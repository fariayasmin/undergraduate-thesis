"""
BPDB Page-3 Dhaka City Substation Dataset Builder

Creates a rerunnable CSV dataset from:
    https://misc.bpdb.gov.bd/daily-generation-archive

What it does:
    1. Crawls the archive table and collects only "Page 3" PDF links.
    2. Downloads/caches PDFs from 2019-01-01 to today.
    3. Parses the Page-3 maximum-load table.
    4. Keeps only Dhaka city substations listed in DHAKA_CITY_SUBSTATIONS.
    5. Writes one CSV row per date, with each substation as a column.

Install:
    pip install requests beautifulsoup4 pdfplumber pandas openpyxl tqdm holidays

Run:
    python3 bpdb_page3_scraper.py

Run again later:
    python3 bpdb_page3_scraper.py

It will crawl the archive again, download newly added PDFs, and refresh recent
dates because BPDB sometimes republishes files shortly after upload.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
import urllib3
import holidays
from bs4 import BeautifulSoup
from tqdm import tqdm


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# #neo4j
# import pandas as pd

# df = pd.read_csv("BPDB_Dhaka_City_Substations_Page3.csv")

# for col in df.columns:
#     if "Load" in col or "Peak" in col:
#         print(col)
        
# print(df[["Aftabnagar_Load_MW", "Aftabnagar_Peak_Time"]].head(10))
# #


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://misc.bpdb.gov.bd"
ARCHIVE_URL = f"{BASE_URL}/daily-generation-archive"

START_DATE = date(2019, 10, 19)
END_DATE = date.today()

PDF_DIR = Path("bpdb_page3_pdfs")
DATA_DIR = Path("data")
OUTPUT_CSV = Path("BPDB_Dhaka_City_Substations_Page3.csv")
OUTPUT_XLSX = Path("BPDB_Dhaka_City_Substations_Page3.xlsx")
URL_CACHE_FILE = Path("bpdb_page3_url_cache.json")
PARSE_CACHE_FILE = Path("bpdb_page3_parse_cache.json")
FAILED_DOWNLOADS_FILE = Path("bpdb_page3_failed_downloads.txt")
FAILED_PARSES_FILE = Path("bpdb_page3_failed_parses.txt")
LOG_FILE = Path("bpdb_page3_scraper.log")
REFERENCE_HOLIDAY_CSV = (
    Path.home()
    / "Downloads"
    / "dhaka_processed_dataset_merged - dhaka_processed_dataset_merged.csv.csv"
)

PDF_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

SSL_VERIFY = False
REQUEST_DELAY_SECONDS = 0.35
MAX_RETRIES = 3
REFRESH_RECENT_DAYS = 45
PARSE_CACHE_VERSION = 6  # bumped — fixed text-strategy cell-merge corruption; old cache has wrong load values

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Substation alias table
# ---------------------------------------------------------------------------
# Each key is the canonical output name.
# The set covers every spelling / OCR variant seen in BPDB's old and new PDFs.
#
# Design notes
# ─────────────
# • "Dhaka University" is abbreviated many ways across PDF generations:
#     D.U. / DU / D U / D.Univ / D.University / Dhaka Univ / Dhaka Uni
#   All are handled.  The bare alias "du" is kept but only matched as a
#   whole token (word-boundary regex), so it won't false-fire on e.g. "Bashundhara".
#
# • "Uttara New" (the second Uttara substation) is also written as:
#     Uttara (New) / New-Uttara / Uttara-New / Uttara 2 / Uttara-2
#   Matching is longest-alias-first so "uttara new" always wins over "uttara".
#
# • Numbered suffixes (-1, -2, -10, I/A, R/A, Industrial) are stripped by the
#   normalise_name function and then matched via the word-boundary containment
#   fallback in canonical_dhaka_substation().
#
# Edit this list if your supervisor defines "Dhaka city" differently.

DHAKA_CITY_SUBSTATIONS: dict[str, set[str]] = {
    # ── Core Dhaka City Corporation (DNCC / DSCC) substations ─────────────
    "Aftabnagar": {
        "aftabnagar", "aftab nagar", "aftab-nagar",
    },
    "Agargaon": {
        "agargaon", "agar gaon",
    },
    "Aminbazar": {
        "aminbazar", "amin bazar", "amin-bazar",
    },
    "Banani": {
        "banani",
    },
    "Banasree": {
        "banasree", "banasri", "bana sri", "banashree",
    },
    "Bangabhaban": {
        "bangabhaban", "banga bhaban", "bangha bhaban",
    },
    "Bashabo": {
        "bashabo", "basabo", "bashabo-1", "basabo-1",
    },
    "Bashundhara": {
        "bashundhara", "basundhara", "bashundhara r/a", "bashundara",
    },
    "Cantonment": {
        "cantonment", "dhaka cantonment", "dhaka cant",
    },
    "Dhanmondi": {
        "dhanmondi", "dhanmondi r/a", "dhanmandi",
    },
    # ── Dhaka University — many abbreviation forms across PDF years ────────
    "Dhaka University": {
        "dhaka university", "dhaka univ", "dhaka uni",
        "d university", "d univ",
        "d u",          # normalised form of D.U. / D U / DU / D.U
        "du",           # bare token — matched at word boundaries only
        "dhakauni",     # seen in some newer PDFs without space
        "dhakauniversity",
    },
    "Gulshan": {
        "gulshan", "gulshan-1", "gulshan-2", "gulshan 1", "gulshan 2",
    },
    "Hasnabad": {
        "hasnabad", "hasan abad", "hasnabad (new)",
    },
    "Hatirjheel": {
        "hatirjheel", "hatir jheel", "hatirjhil",
    },
    "Kalyanpur": {
        "kalyanpur", "kallayanpur", "kallyanpur", "kalyanpur-1",
    },
    "Kamrangirchar": {
        "kamrangirchar", "kamrangir char", "kamrangirchar (new)",
    },
    "Keranigonj": {
        "keranigonj", "keraniganj", "kerani gonj", "kerani ganj",
    },
    "Kodda": {
        "kodda",                    # Badda-Khilgaon corridor, east Dhaka
    },
    "Lalbag": {
        "lalbag", "lalbagh", "lal bag", "lal bagh",
    },
    "Madartek": {
        "madartek", "madar tek",    # Khilgaon, inside DSCC
    },
    "Maniknagar": {
        "maniknagar", "manik nagar", "manik-nagar",
    },
    "Matual": {
        "matual", "matuail", "matual (new)",
    },
    # ── Dhaka Metro Rail substations (MRT Line 6) ─────────────────────────
    "Metrorail Motijheel": {
        "metrorail motijheel", "metrorail_motijheel",
        "metro motijheel", "mrt motijheel",
    },
    "Metrorail Uttara": {
        "metrorail uttara", "metrorail_uttara",
        "metro uttara", "mrt uttara",
    },
    "Mirpur": {
        "mirpur",
        "mirpur-1", "mirpur-2", "mirpur-10",
        "mirpur 1", "mirpur 2", "mirpur 10",
    },
    "Moghbazar": {
        "moghbazar", "mogbazar", "mogh bazar", "mog bazar", "moghbazar-1",
        "magbazar", "mag bazar",   # 2019-era PDF uses 'Mag' not 'Mog'
    },
    "Motijheel": {
        "motijheel", "motijhil", "moti jheel", "moti-jheel",
    },
    "Narinda": {
        "narinda", "narinda-1",
    },
    "Postogola": {
        "postogola", "postagola", "postgola",
    },
    "Pubail": {
        "pubail",                   # northeast Dhaka fringe / Rupganj border
    },
    "Purbachal": {
        "purbachal", "purba chal", "purbachal new town",
    },
    "Rampura": {
        "rampura", "ram pura", "ram-pura",
    },
    "Savar": {
        "savar",                    # Dhaka district EPZ / DEPZ corridor
    },
    "Shyampur": {
        "shyampur", "shampur", "shyam pur", "sham pur", "shyampur-1",
    },
    "Tejgaon": {
        "tejgaon", "tejgoan",
        "tejgaon i/a", "tejgaon industrial", "tejgaon ia",
    },
    "Ullon": {
        "ullon",                    # Demra / Shyampur corridor, south-east Dhaka
    },
    "Uttara": {
        "uttara", "uttara-1", "uttara 1",
    },
    # ── Must appear AFTER "Uttara" so longest-alias-first wins ────────────
    "Uttara New": {
        "uttara new", "new uttara",
        "uttara (new)", "new-uttara", "uttara-new",
        "uttara 2", "uttara-2", "uttaranew",
    },
}

SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Pre_Monsoon",
    4: "Pre_Monsoon",
    5: "Pre_Monsoon",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Post_Monsoon",
    11: "Post_Monsoon",
}
SEASON_ENC = {
    "Winter": 0,
    "Pre_Monsoon": 1,
    "Monsoon": 2,
    "Post_Monsoon": 3,
}
HOLIDAY_NAME_MAP = {
    "eid al fitr": "Eid-ul-Fitr",
    "eid al adha": "Eid-ul-Adha",
    "victory day": "Victory Day",
    "christmas day": "Christmas",
    "christmas": "Christmas",
    "martyrs day and international mother language day": "International mother language day",
    "international mother language day": "International mother language day",
    "sheikh mujibur rahman s birthday": "Father of nation's Birth",
    "independence day": "Independence Day",
    "mid sha ban": "Sab-E-Barat",
    "shab e barat": "Sab-E-Barat",
    "pahela baishakh": "Pohela Baishakh",
    "pohela baishakh": "Pohela Baishakh",
    "may day": "May Day",
    "buddha purnima": "Buddho Purnima",
    "janmashtami": "Janmo-astami",
    "national mourning day": "Jatiyo shokh dibosh",
    "durga puja": "Durga Puja",
    "ashura": "Ashura",
    "mawlid": "EID-E-MILADUNNOBI",
    "eid e milad": "EID-E-MILADUNNOBI",
    "laylat al qadr": "Ramadan+Sab-E-QADAR",
    "jumu atul wida": "Ramadan",
}

# The `holidays` library for Bangladesh returns names in Bengali script.
# normalise_name() strips all non-ASCII, leaving "". We map the raw Bengali
# strings directly to canonical English names before any further processing.
BENGALI_TO_ENGLISH: dict[str, str] = {
    "ঈদুল ফিতর":       "Eid-ul-Fitr",
    "ঈদুল আজহা":       "Eid-ul-Adha",
    "শবে বরাত":        "Sab-E-Barat",
    "শবে কদর":         "Ramadan+Sab-E-QADAR",
    "জুমাতুল বিদা":    "Ramadan",
    "জুমাতুল বিদা; শবে কদর": "Ramadan+Sab-E-QADAR",
    "পহেলা বৈশাখ":     "Pohela Baishakh",
    "মে দিবস":         "May Day",
    "আশুরা":           "Ashura",
    "ঈদে মিলাদুন্নবী": "EID-E-MILADUNNOBI",
    "বিজয় দিবস":      "Victory Day",
    "বড়দিন":          "Christmas",
    "জাতীয় শোক দিবস": "Jatiyo shokh dibosh",
    "স্বাধীনতা দিবস":  "Independence Day",
    "শহীদ দিবস ও আন্তর্জাতিক মাতৃভাষা দিবস": "International mother language day",
    "জাতির পিতা বঙ্গবন্ধু শেখ মুজিবুর রহমান এর জন্মদিবস": "Father of nation's Birth",
    "সাধারণ ছুটি":     "Public Holiday",
    "বুদ্ধ পূর্ণিমা":  "Buddho Purnima",
    "জন্মাষ্টমী":      "Janmo-astami",
    "দুর্গাপূজা":      "Durga Puja",
}

# ---------------------------------------------------------------------------
# Gazette-declared holiday blocks — overrides the `holidays` library where
# the Bangladesh government declared more/fewer days than the Islamic
# calendar minimum. The library always gives exactly 3 days for each Eid;
# the actual gazette varies from 3 to 5 days depending on the year and
# whether adjacent weekends are bridged.
#
# Sources: Bangladesh Gazette notifications (bdjobs.com/holiday, BPDB notices)
# Durga Puja is entirely absent from the library so all years are listed here.
#
# Key   = date
# Value = (holiday_name, holiday_type)
# ---------------------------------------------------------------------------
GAZETTE_HOLIDAYS: dict[date, tuple[str, str]] = {

    # ── Eid-ul-Fitr ────────────────────────────────────────────────────────
    # 2019: Mar 31 – Apr 5 (6 days — Fri/Sat weekend bridged both sides)
    date(2019, 6, 3):  ("Eid-ul-Fitr", "Eid"),
    date(2019, 6, 4):  ("Eid-ul-Fitr", "Eid"),
    date(2019, 6, 5):  ("Eid-ul-Fitr", "Eid"),
    date(2019, 6, 6):  ("Eid-ul-Fitr", "Eid"),
    date(2019, 6, 7):  ("Eid-ul-Fitr", "Eid"),

    # 2020: 3 days (library correct)
    date(2020, 5, 22): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2020, 5, 23): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2020, 5, 24): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2020, 5, 25): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2020, 5, 26): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2020, 5, 27): ("Eid-ul-Fitr", "Eid"),   # extra day before

    # 2021: 3 days (library correct — COVID restrictions reduced public holidays)
    date(2021, 5, 12): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2021, 5, 13): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2021, 5, 14): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2021, 5, 15): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2021, 5, 16): ("Eid-ul-Fitr", "Eid"),   # extra day before

    # 2022: 3 days (library correct)
    date(2022, 4, 29): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 30): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 5, 1): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 5, 2): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 5, 3): ("Eid-ul-Fitr", "Eid"),   # extra day before

    # 2023: 3 days (library correct)
    date(2023, 4, 21): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 22): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 23): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 24): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 25): ("Eid-ul-Fitr", "Eid"),   # extra day before


    # 2024: 3 days (library correct)
    date(2024, 4, 10): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 11): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 12): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 13): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2024, 4, 14): ("Eid-ul-Fitr", "Eid"),   # extra day before


    # 2025: 3 days (library correct Mar 31 – Apr 2)
    date(2025, 3, 28): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2025, 3, 29): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2025, 3, 30): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2025, 3, 31): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2025, 4, 1): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2025, 4, 2): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2025, 4, 3): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2025, 4, 4): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2025, 4, 5): ("Eid-ul-Fitr", "Eid"),   # extra day before

    # 2026: 5 days — Mar 19 (Thu) to Mar 23 (Mon); library only has Mar 21–23
    date(2026, 3, 17): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2026, 3, 18): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2026, 3, 19): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2026, 3, 20): ("Eid-ul-Fitr", "Eid"),   # extra day before
    date(2026, 3, 21): ("Eid-ul-Fitr", "Eid"),   # library has this
    date(2026, 3, 22): ("Eid-ul-Fitr", "Eid"),   # library has this
    date(2026, 3, 23): ("Eid-ul-Fitr", "Eid"),   # library has this

    # ── Eid-ul-Adha ────────────────────────────────────────────────────────
    # Add extended blocks here as gazette notifications are published
    # ── Eid-ul-Adha ────────────────────────────────────────────────────────

    # 2019: 9–13 Aug (5 days)
    date(2019, 8, 9):  ("Eid-ul-Adha", "Eid"),
    date(2019, 8, 10): ("Eid-ul-Adha", "Eid"),
    date(2019, 8, 11): ("Eid-ul-Adha", "Eid"),
    date(2019, 8, 12): ("Eid-ul-Adha", "Eid"),
    date(2019, 8, 13): ("Eid-ul-Adha", "Eid"),

    # 2020: 30 Jul – 3 Aug (5 days)
    date(2020, 7, 30): ("Eid-ul-Adha", "Eid"),
    date(2020, 7, 31): ("Eid-ul-Adha", "Eid"),
    date(2020, 8, 1):  ("Eid-ul-Adha", "Eid"),
    date(2020, 8, 2):  ("Eid-ul-Adha", "Eid"),
    date(2020, 8, 3):  ("Eid-ul-Adha", "Eid"),

    # 2021: 19–23 Jul (5 days)
    date(2021, 7, 19): ("Eid-ul-Adha", "Eid"),
    date(2021, 7, 20): ("Eid-ul-Adha", "Eid"),
    date(2021, 7, 21): ("Eid-ul-Adha", "Eid"),
    date(2021, 7, 22): ("Eid-ul-Adha", "Eid"),
    date(2021, 7, 23): ("Eid-ul-Adha", "Eid"),

    # 2022: 8–12 Jul (5 days)
    date(2022, 7, 8):  ("Eid-ul-Adha", "Eid"),
    date(2022, 7, 9):  ("Eid-ul-Adha", "Eid"),
    date(2022, 7, 10): ("Eid-ul-Adha", "Eid"),
    date(2022, 7, 11): ("Eid-ul-Adha", "Eid"),
    date(2022, 7, 12): ("Eid-ul-Adha", "Eid"),

    # 2023: 27 Jun – 1 Jul (5 days)
    date(2023, 6, 27): ("Eid-ul-Adha", "Eid"),
    date(2023, 6, 28): ("Eid-ul-Adha", "Eid"),
    date(2023, 6, 29): ("Eid-ul-Adha", "Eid"),
    date(2023, 6, 30): ("Eid-ul-Adha", "Eid"),
    date(2023, 7, 1):  ("Eid-ul-Adha", "Eid"),

    # 2024: 15–19 Jun (5 days)
    date(2024, 6, 15): ("Eid-ul-Adha", "Eid"),
    date(2024, 6, 16): ("Eid-ul-Adha", "Eid"),
    date(2024, 6, 17): ("Eid-ul-Adha", "Eid"),
    date(2024, 6, 18): ("Eid-ul-Adha", "Eid"),
    date(2024, 6, 19): ("Eid-ul-Adha", "Eid"),

    # 2025: 5–14 Jun (10 days, gazette)
    date(2025, 6, 5):  ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 6):  ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 7):  ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 8):  ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 9):  ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 10): ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 11): ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 12): ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 13): ("Eid-ul-Adha", "Eid"),
    date(2025, 6, 14): ("Eid-ul-Adha", "Eid"),

    # 2026: 25–31 May (7 days)
    date(2026, 5, 25): ("Eid-ul-Adha", "Eid"),
    date(2026, 5, 26): ("Eid-ul-Adha", "Eid"),
    date(2026, 5, 27): ("Eid-ul-Adha", "Eid"),
    date(2026, 5, 28): ("Eid-ul-Adha", "Eid"),
    date(2026, 5, 29): ("Eid-ul-Adha", "Eid"),
    date(2026, 5, 30): ("Eid-ul-Adha", "Eid"),
    date(2026, 5, 31): ("Eid-ul-Adha", "Eid"),
    
    # ── Durga Puja — 3 days every year, entirely missing from library ──────
    date(2019, 10, 8):  ("Durga Puja",           "Durga Puja"),
    date(2020, 10, 26): ("Durga Puja",           "Durga Puja"),
    date(2021, 10, 15): ("Durga Puja ",  "Durga Puja"),
    date(2021, 10, 16): ("Durga Puja",           "Durga Puja"),
    date(2022, 10, 5):  ("Durga Puja",           "Durga Puja"),
    date(2023, 10, 24): ("Durga Puja",           "Durga Puja"),
    date(2024, 10, 11): ("Durga Puja ",  "Durga Puja"),
    date(2024, 10, 12): ("Durga Puja",           "Durga Puja"),
    date(2024, 10, 13): ("Durga Puja",           "Durga Puja"),
    date(2025, 10, 1):  ("Durga Puja",           "Durga Puja"),
    date(2025, 10, 2):  ("Durga Puja",           "Durga Puja"),
    date(2025, 10, 3):  ("Durga Puja",           "Durga Puja"),
    date(2025, 10, 4):  ("Durga Puja",           "Durga Puja"),
    date(2026, 10, 20): ("Durga Puja ",  "Durga Puja"),
    date(2026, 10, 21): ("Durga Puja",           "Durga Puja"),
}
# Bangladesh weekend is Friday/Saturday (changed from Fri/Sun in 2017).
# Used for Is_Weekend.
WEEKEND_DAYS = {4, 5}  # Mon=0 ... Fri=4, Sat=5 (per datetime.weekday())

# ---------------------------------------------------------------------------
# Weather (Open-Meteo Historical Weather API — ERA5 reanalysis, free, no key,
# CC BY 4.0). Single point used to represent "Dhaka city" weather since the
# load dataset is at city scale, not per-substation. Coordinates are central
# Dhaka (near Motijheel/old town); ERA5 resolution is ~25km so this is fine
# for a city-level daily feature, not meant to capture neighborhood variation.
# https://open-meteo.com/en/docs/historical-weather-api
# ---------------------------------------------------------------------------
WEATHER_LATITUDE = 23.8103
WEATHER_LONGITUDE = 90.4125
WEATHER_TIMEZONE = "Asia/Dhaka"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_CACHE_FILE = Path("bpdb_page3_weather_cache.json")
WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("bpdb_page3")


@dataclass(frozen=True)
class ArchiveEntry:
    report_date: date
    page3_url: str


def normalise_name(value: str) -> str:
    """
    Lower-case, strip punctuation (keeping digits), collapse whitespace.

    Examples
    --------
    "D.U."          -> "d u"
    "Tejgaon I/A"   -> "tejgaon i a"   (then alias "tejgaon ia" matches)
    "Mirpur-10"     -> "mirpur 10"     (suffix handled by containment regex)
    "Bashundhara R/A" -> "bashundhara r a"
    """
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# Build the lookup dict once at import time.
# Sorted longest-alias-first so "uttara new" always wins over "uttara",
# and "dhaka university" always wins over "du".
ALIAS_TO_CANONICAL: dict[str, str] = {
    normalise_name(alias): canonical
    for canonical, aliases in DHAKA_CITY_SUBSTATIONS.items()
    for alias in aliases | {canonical}
}

# Pre-sorted list used by the containment-fallback loop (avoids re-sorting each call).
_ALIASES_LONGEST_FIRST: list[tuple[str, str]] = sorted(
    ALIAS_TO_CANONICAL.items(), key=lambda kv: -len(kv[0])
)

# Tokens that are never a substation name (guards against header rows / totals).
_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "sub station", "substation", "station",
        "total", "sl no", "sl", "no",
        "load mw", "load", "time", "hour",
        "name", "s n", "sn",
    }
)


def canonical_dhaka_substation(raw_name: str) -> str | None:
    """
    Map any raw PDF cell value to a canonical substation name, or return None.

    Strategy (in order):
    1. Exact lookup on the normalised alias table.
    2. Longest-alias-first word-boundary containment search — handles suffixes
       like "-1", "-10", "I/A", "R/A", "(New)", "Industrial".
    """
    clean = normalise_name(raw_name)
    if not clean or clean in _STOP_TOKENS:
        return None

    # 1 — exact match
    if clean in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[clean]

    # 2 — containment: longest alias first so "uttara new" beats "uttara"
    for alias, canonical in _ALIASES_LONGEST_FIRST:
        # \b-style: alias must start/end at a word boundary or digit boundary
        pattern = rf"(^|\s){re.escape(alias)}($|\s|\d)"
        if re.search(pattern, clean):
            return canonical

    return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    text = re.sub(r"[^\d.]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_report_date(text: str) -> date | None:
    patterns = [
        r"\bDate\s*:?\s*(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b",
        r"\bDate\s*:?\s*(\d{1,2})[-\s]([A-Za-z]{3,9})[-\s](\d{2,4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        d, m, y = match.groups()
        year = int(y)
        if year < 100:
            year += 2000
        try:
            if m.isdigit():
                return date(year, int(m), int(d))
            month = datetime.strptime(m[:3].title(), "%b").month
            return date(year, month, int(d))
        except ValueError:
            return None
    return None


def archive_date_from_text(raw: str) -> date | None:
    raw = raw.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d-%b-%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    return None


def request_text(url: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=35,
                verify=SSL_VERIFY,
            )
            if response.status_code == 404:
                return None
            if response.status_code == 200:
                return response.text
            log.warning("HTTP %s for %s", response.status_code, url)
        except requests.RequestException as exc:
            log.warning("Request failed for %s: %s", url, exc)
        if attempt < MAX_RETRIES:
            time.sleep(attempt * 2)
    return None


def download_pdf(url: str, destination: Path) -> bool:
    temp_destination = destination.with_suffix(".part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(
                url,
                headers=HEADERS,
                timeout=60,
                stream=True,
                verify=SSL_VERIFY,
            ) as response:
                if response.status_code == 404:
                    return False
                if response.status_code != 200:
                    log.warning("HTTP %s while downloading %s", response.status_code, url)
                else:
                    first = b""
                    with temp_destination.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 64):
                            if not chunk:
                                continue
                            if not first:
                                first = chunk
                            handle.write(chunk)

                    if first[:4] != b"%PDF" or temp_destination.stat().st_size < 500:
                        temp_destination.unlink(missing_ok=True)
                        return False

                    temp_destination.replace(destination)
                    return True
        except requests.RequestException as exc:
            log.warning("Download failed for %s: %s", url, exc)
        finally:
            temp_destination.unlink(missing_ok=True)

        if attempt < MAX_RETRIES:
            time.sleep(attempt * 2)
    return False


def parse_archive_page(html: str) -> list[ArchiveEntry]:
    soup = BeautifulSoup(html, "html.parser")
    entries: list[ArchiveEntry] = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        report_date = archive_date_from_text(cells[1].get_text(" ", strip=True))
        if report_date is None:
            continue

        page3_link = cells[4].find("a", href=True)
        if page3_link is None:
            continue

        page3_url = urljoin(BASE_URL, page3_link["href"].strip())
        entries.append(ArchiveEntry(report_date=report_date, page3_url=page3_url))

    return entries


def max_archive_page_number(html: str) -> int | None:
    soup = BeautifulSoup(html, "html.parser")
    page_numbers = []
    for link in soup.find_all("a", href=True):
        match = re.search(r"[?&]page=(\d+)", link["href"])
        if match:
            page_numbers.append(int(match.group(1)))
            continue
        text = link.get_text(" ", strip=True)
        if text.isdigit():
            page_numbers.append(int(text))
    return max(page_numbers) if page_numbers else None


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def load_parse_cache() -> dict[str, list[dict]]:
    raw_cache = load_json(PARSE_CACHE_FILE, {})
    if not raw_cache:
        return {}

    if raw_cache.get("_cache_version") != PARSE_CACHE_VERSION:
        log.info("Ignoring old parse cache (version mismatch); PDFs will be reparsed.")
        return {}

    return raw_cache.get("records", {})


def save_parse_cache(records: dict[str, list[dict]]) -> None:
    save_json(
        PARSE_CACHE_FILE,
        {
            "_cache_version": PARSE_CACHE_VERSION,
            "records": records,
        },
    )


def build_url_cache(start_date: date, end_date: date) -> dict[str, str]:
    cache: dict[str, str] = load_json(URL_CACHE_FILE, {})
    needed_dates = {
        current.date().isoformat()
        for current in pd.date_range(start_date, end_date)
    }

    log.info("Crawling BPDB archive for Page-3 PDF links")
    empty_pages = 0
    page_number = 1
    max_page_number: int | None = None

    while empty_pages < 3 and (max_page_number is None or page_number <= max_page_number):
        html = request_text(f"{ARCHIVE_URL}?page={page_number}")
        if html is None:
            empty_pages += 1
            page_number += 1
            continue

        if page_number == 1:
            max_page_number = max_archive_page_number(html)
            if max_page_number:
                log.info("Archive reports %s pages", max_page_number)

        entries = parse_archive_page(html)
        if not entries:
            empty_pages += 1
            page_number += 1
            continue

        empty_pages = 0
        for entry in entries:
            if start_date <= entry.report_date <= end_date:
                key = entry.report_date.isoformat()
                cache[key] = entry.page3_url

        if page_number % 25 == 0:
            save_json(URL_CACHE_FILE, cache)
            log.info("Archive checkpoint: page %s, cached URLs %s", page_number, len(cache))

        oldest_on_page = min(entry.report_date for entry in entries)
        if oldest_on_page < start_date and needed_dates.issubset(set(cache)):
            break

        page_number += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    save_json(URL_CACHE_FILE, cache)
    log.info("URL cache contains %s Page-3 links", len(cache))
    return cache


def local_pdf_path(date_str: str) -> Path:
    return PDF_DIR / f"page3_{date_str}.pdf"


def download_missing_pdfs(
    url_cache: dict[str, str],
    start_date: date,
    end_date: date,
    refresh_recent_days: int,
) -> list[str]:
    refresh_from = end_date - timedelta(days=refresh_recent_days)
    to_download: list[tuple[str, str]] = []

    for date_str, url in sorted(url_cache.items()):
        report_date = date.fromisoformat(date_str)
        if not (start_date <= report_date <= end_date):
            continue
        path = local_pdf_path(date_str)
        should_refresh = report_date >= refresh_from
        if should_refresh or not path.exists() or path.stat().st_size < 500:
            to_download.append((date_str, url))

    if not to_download:
        log.info("No PDF downloads needed")
        return []
    failed: list[str] = []
    for date_str, url in tqdm(to_download, desc="Downloading Page-3 PDFs"):
        if not download_pdf(url, local_pdf_path(date_str)):
            failed.append(date_str)
        time.sleep(REQUEST_DELAY_SECONDS)

    if failed:
        FAILED_DOWNLOADS_FILE.write_text("\n".join(failed), encoding="utf-8")
    log.info("Downloaded/refreshed %s PDFs; failed %s", len(to_download) - len(failed), len(failed))
    return failed


def clean_cell(value) -> str:
    if value is None:
        return ""
    value = str(value).replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def looks_like_time(value: str) -> bool:
    # Accepts HH:MM (new PDFs), HH-MM (old PDFs), HH.MM, and bare HHMM.
    # Old BPDB PDFs write peak hour as "18-30", "00-00" etc. with a dash
    # separator — the dash form was previously unrecognised, causing the
    # time cell to be silently ignored and the guard to misfire on real loads.
    value = value.strip()
    return bool(re.fullmatch(r"\d{1,2}[:\-\.]\d{2}", value)) or bool(re.fullmatch(r"\d{3,4}", value))


def normalise_time(value: str) -> str | None:
    # Normalise HH-MM (old PDFs), HH.MM, and HH:MM all to "HH:MM".
    value = value.strip().replace(".", ":").replace("-", ":")
    if re.fullmatch(r"\d{3,4}", value):
        value = value.zfill(4)
        value = f"{value[:2]}:{value[2:]}"
    if not re.fullmatch(r"\d{1,2}:\d{2}", value):
        return None
    hour, minute = value.split(":")
    try:
        h = int(hour)
        m = int(minute)
    except ValueError:
        return None
    if not (0 <= h <= 24 and 0 <= m <= 59):
        return None
    if h == 24:
        h = 0
    return f"{h:02d}:{m:02d}"


def record_from_triplet(
    name: str,
    load_value: str,
    time_value: str,
    report_date: date,
    source_pdf: Path,
    source_url: str | None,
) -> dict | None:
    canonical = canonical_dhaka_substation(name)
    if canonical is None:
        return None

    load_mw = parse_float(load_value)
    if load_mw is None or not (0 <= load_mw <= 2000):
        return None

    peak_time = normalise_time(time_value) if time_value else None
    return {
        "Date": report_date.isoformat(),
        "Substation": canonical,
        "Load_MW": load_mw,
        "Time": peak_time,
        "Raw_Substation": clean_cell(name),
        "Source_PDF": source_pdf.name,
        "Source_URL": source_url,
    }


def extract_records_from_table(
    table: list[list],
    report_date: date,
    source_pdf: Path,
    source_url: str | None,
) -> list[dict]:
    records: list[dict] = []
    for row in table:
        cells = [clean_cell(cell) for cell in row]
        if len(cells) < 3:
            continue

        lower_row = " ".join(cells).lower()
        if "sub-station" in lower_row or "substation" in lower_row:
            continue

        # Works for both formats:
        #   old: Sub-station | Load (MW) | Hour | Sub-station | Load | Hour
        #   new: Sl No. | Sub-station | Load (MW) | Time | Sl No. | ...
        for index in range(0, len(cells) - 2):
            name = cells[index]
            load_value = cells[index + 1]
            time_value = cells[index + 2]

            if parse_float(load_value) is None:
                continue
            if time_value and not looks_like_time(time_value):
                continue

            record = record_from_triplet(
                name=name,
                load_value=load_value,
                time_value=time_value,
                report_date=report_date,
                source_pdf=source_pdf,
                source_url=source_url,
            )
            if record:
                records.append(record)
    return records


def extract_records_from_text(
    text: str,
    report_date: date,
    source_pdf: Path,
    source_url: str | None,
) -> list[dict]:
    """
    Regex fallback when pdfplumber table extraction yields nothing.

    Scans each line for a known substation alias, then grabs the first
    numeric value after it as the load, and an optional HH:MM / HHMM
    time after that.
    """
    records: list[dict] = []

    # Build alias list sorted longest-first so multi-word aliases match before
    # their prefixes (e.g. "dhaka university" before "dhaka").
    aliases = [alias for alias, _ in _ALIASES_LONGEST_FIRST]

    for line in text.splitlines():
        clean_line = re.sub(r"\s+", " ", line).strip()
        normal_line = normalise_name(clean_line)
        if not clean_line:
            continue

        for alias in aliases:
            if not re.search(rf"(^|\s){re.escape(alias)}($|\s|\d)", normal_line):
                continue

            canonical = ALIAS_TO_CANONICAL[alias]
            raw_alias = re.escape(alias).replace(r"\ ", r"\s+")
            match = re.search(rf"{raw_alias}\D+(\d+(?:\.\d+)?)", clean_line, flags=re.IGNORECASE)
            if not match:
                continue

            load_mw = parse_float(match.group(1))
            if load_mw is None or not (0 <= load_mw <= 2000):
                continue

            time_match = re.search(r"\b(\d{1,2}[:\-\.]\d{2}|\d{3,4})\b", clean_line[match.end():])

            records.append(
                {
                    "Date": report_date.isoformat(),
                    "Substation": canonical,
                    "Load_MW": load_mw,
                    "Time": normalise_time(time_match.group(1)) if time_match else None,
                    "Raw_Substation": canonical,
                    "Source_PDF": source_pdf.name,
                    "Source_URL": source_url,
                }
            )
            break

    return records


def _table_looks_valid(records: list[dict]) -> bool:
    """
    Return True only if the table produced at least one record where BOTH
    load and time are populated and the load is in a physically realistic
    range for a single Dhaka substation (5–500 MW).

    The text-strategy table on old PDFs merges 'Substation Load' into one
    cell, producing spurious records like Shyampur=1800 MW with no time.
    We reject such outputs here so the code falls through to the reliable
    line-by-line text extractor.
    """
    good = [
        r for r in records
        if r.get("Time") is not None and 5 <= r.get("Load_MW", 0) <= 500
    ]
    return len(good) >= 3   # at least 3 plausible Dhaka records


def parse_pdf(pdf_path: Path, fallback_date: date, source_url: str | None) -> list[dict]:
    all_text_parts: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            all_text_parts.append(text)

        report_date = parse_report_date("\n".join(all_text_parts)) or fallback_date

        # ── Try table extraction strategies in order ───────────────────────
        # The "lines" strategy works well on new PDFs (2023+) which have
        # proper ruled borders. The "text" strategy is tried next but on
        # old PDFs (2019-2022) it merges "Substation Load" into one cell,
        # producing garbage. We validate each strategy's output before
        # accepting it; if neither yields clean data we fall through to the
        # line-by-line text extractor which handles old PDFs correctly.
        records: list[dict] = []
        for settings in (
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "intersection_tolerance": 5,
            },
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
                "join_tolerance": 3,
            },
        ):
            candidate: list[dict] = []
            for page in pdf.pages:
                try:
                    tables = page.extract_tables(table_settings=settings) or []
                except Exception:
                    tables = []
                for table in tables:
                    candidate.extend(
                        extract_records_from_table(table, report_date, pdf_path, source_url)
                    )
            if candidate and _table_looks_valid(candidate):
                records = candidate
                break   # good table found; skip remaining strategies

    # ── Text-line fallback ─────────────────────────────────────────────────
    # Used when no table strategy produced valid data (common for 2019-2022
    # PDFs where the main data table has no ruled borders and pdfplumber's
    # text strategy merges cells incorrectly).
    if not records:
        records = extract_records_from_text(
            "\n".join(all_text_parts), report_date, pdf_path, source_url
        )

    # Deduplicate: keep last occurrence of each (date, substation) pair.
    deduped: dict[tuple[str, str], dict] = {}
    for record in records:
        deduped[(record["Date"], record["Substation"])] = record
    return list(deduped.values())


def _parse_one(args: tuple) -> tuple[str, list[dict]]:
    """
    Top-level function (required for multiprocessing pickling).
    Returns (date_key, records).  Never raises — errors become an empty list.
    """
    date_key, pdf_path_str, source_url = args
    pdf_path = Path(pdf_path_str)
    fallback_date = date.fromisoformat(date_key)
    try:
        records = parse_pdf(pdf_path, fallback_date, source_url)
        return date_key, records
    except Exception as exc:
        return date_key, []  # logged below after collection


def parse_all_pdfs(
    url_cache: dict[str, str],
    start_date: date,
    end_date: date,
    refresh_recent_days: int,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Parse all downloaded PDFs, using `workers` parallel processes.

    The parse cache is checked first; only PDFs that are new or fall within
    the refresh window are sent to the worker pool.  Results are flushed to
    the cache every 100 completed PDFs so progress is not lost on interruption.
    """
    parse_cache: dict[str, list[dict]] = load_parse_cache()
    refresh_from = end_date - timedelta(days=refresh_recent_days)
    failed: list[str] = []

    # ── Collect PDFs that need (re)parsing ────────────────────────────────
    pdfs = sorted(PDF_DIR.glob("page3_*.pdf"))
    todo: list[tuple[str, str, str | None]] = []   # (date_key, path_str, url)
    for pdf_path in pdfs:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", pdf_path.stem)
        if not match:
            continue
        report_date = date.fromisoformat(match.group(1))
        if not (start_date <= report_date <= end_date):
            continue
        date_key = report_date.isoformat()
        should_refresh = report_date >= refresh_from
        if date_key in parse_cache and not should_refresh:
            continue
        todo.append((date_key, str(pdf_path), url_cache.get(date_key)))

    if not todo:
        log.info("Parse cache is up to date — no PDFs need parsing.")
    else:
        log.info(
            "Parsing %s PDFs with %s worker%s …",
            len(todo), workers, "s" if workers > 1 else "",
        )
        flush_every = 100
        completed = 0

        if workers <= 1:
            # Single-process path (safe in all environments, e.g. Colab notebooks)
            for args in tqdm(todo, desc="Parsing PDFs"):
                date_key, records = _parse_one(args)
                if not records and date_key not in {a[0] for a in todo if a[0] == date_key}:
                    failed.append(date_key)
                parse_cache[date_key] = records
                completed += 1
                if completed % flush_every == 0:
                    save_parse_cache(parse_cache)
        else:
            # Multiprocessing path
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for date_key, records in tqdm(
                    pool.imap_unordered(_parse_one, todo, chunksize=4),
                    total=len(todo),
                    desc=f"Parsing PDFs ({workers} workers)",
                ):
                    if not records:
                        failed.append(date_key)
                        log.debug("Empty result for %s", date_key)
                    parse_cache[date_key] = records
                    completed += 1
                    if completed % flush_every == 0:
                        save_parse_cache(parse_cache)

        save_parse_cache(parse_cache)
        log.info("Parsed %s PDFs; %s returned no records.", len(todo), len(failed))

    # ── Assemble rows from cache ───────────────────────────────────────────
    rows = []
    for date_key, records in parse_cache.items():
        report_date = date.fromisoformat(date_key)
        if start_date <= report_date <= end_date:
            rows.extend(records)

    if failed:
        FAILED_PARSES_FILE.write_text("\n".join(sorted(failed)), encoding="utf-8")

    if not rows:
        return pd.DataFrame(
            columns=["Date", "Substation", "Load_MW", "Time", "Raw_Substation", "Source_PDF", "Source_URL"]
        )

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Load_MW"] = pd.to_numeric(df["Load_MW"], errors="coerce")
    df = df.dropna(subset=["Date", "Substation", "Load_MW"])
    df = df.drop_duplicates(subset=["Date", "Substation"], keep="last")
    df = df.sort_values(["Date", "Substation"]).reset_index(drop=True)
    return df


def load_reference_holidays() -> dict[date, str]:
    if not REFERENCE_HOLIDAY_CSV.exists():
        return {}
    try:
        ref = pd.read_csv(
            REFERENCE_HOLIDAY_CSV,
            usecols=["Date", "Holiday name"],
        )
    except Exception as exc:
        log.warning("Could not read reference holiday CSV: %s", exc)
        return {}

    ref["Date"] = pd.to_datetime(ref["Date"], errors="coerce")
    ref = ref.dropna(subset=["Date"])
    return {
        row["Date"].date(): str(row["Holiday name"])
        for _, row in ref.iterrows()
    }


def standard_holiday_name(raw_name: str) -> str:
    # Direct Bengali → English lookup first (holidays library returns Bengali).
    if raw_name in BENGALI_TO_ENGLISH:
        return BENGALI_TO_ENGLISH[raw_name]
    # Fallback: normalise and substring-match the English map.
    normalised = normalise_name(raw_name)
    for key, value in HOLIDAY_NAME_MAP.items():
        if key in normalised:
            return value
    return raw_name

def _is_weekend(d: date) -> bool:
    """Bangladesh weekend: Friday & Saturday."""
    return d.weekday() in WEEKEND_DAYS


_WEATHER_COLUMNS = [
    "Temp_Max_C",
    "Temp_Min_C",
    "Temp_Mean_C",
    "Humidity_Mean_Pct",
    "Precip_Sum_mm",
    "Rain_Sum_mm",
    "Wind_Max_kmh",
]


def _request_weather_chunk(start: date, end: date) -> dict | None:
    params = {
        "latitude": WEATHER_LATITUDE,
        "longitude": WEATHER_LONGITUDE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(WEATHER_DAILY_VARS),
        "timezone": WEATHER_TIMEZONE,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                WEATHER_ARCHIVE_URL,
                params=params,
                timeout=35,
                verify=SSL_VERIFY,
            )
            if response.status_code == 200:
                return response.json().get("daily", {})
            log.warning("Weather API HTTP %s for %s to %s", response.status_code, start, end)
        except requests.RequestException as exc:
            log.warning("Weather request failed for %s to %s: %s", start, end, exc)
        except ValueError as exc:
            log.warning("Weather API returned non-JSON for %s to %s: %s", start, end, exc)
        if attempt < MAX_RETRIES:
            time.sleep(attempt * 2)
    return None


def fetch_weather_context(dates: pd.Series) -> pd.DataFrame:
    """
    Fetch daily weather (Dhaka, ERA5 reanalysis) for the given dates via the
    free Open-Meteo Historical Weather API. Results are cached to disk by
    date so re-runs don't re-fetch already-known days, and "today" (which has
    no finalized ERA5 value yet) is always re-fetched.

    Returns a DataFrame with: Date, Temp_Max_C, Temp_Min_C, Temp_Mean_C,
    Humidity_Mean_Pct, Precip_Sum_mm, Rain_Sum_mm, Wind_Max_kmh.
    On total fetch failure (e.g. no network), returns the same columns filled
    with NA rather than raising, so the rest of the pipeline still runs.
    """
    all_dates = sorted(set(pd.to_datetime(dates).dt.date))
    if not all_dates:
        return pd.DataFrame(columns=["Date", *_WEATHER_COLUMNS])

    cache: dict[str, dict] = load_json(WEATHER_CACHE_FILE, {})
    today = date.today()
    missing = [d for d in all_dates if d.isoformat() not in cache or d == today]

    if missing:
        log.info("Fetching weather for %s day(s) from Open-Meteo", len(missing))
        # Open-Meteo accepts an arbitrary start/end range per request; fetch
        # one contiguous request per missing block instead of per-day calls.
        missing_sorted = sorted(missing)
        chunk_start = missing_sorted[0]
        chunk_end = missing_sorted[0]
        chunks: list[tuple[date, date]] = []
        for d in missing_sorted[1:]:
            if (d - chunk_end).days <= 7:
                chunk_end = d
            else:
                chunks.append((chunk_start, chunk_end))
                chunk_start = chunk_end = d
        chunks.append((chunk_start, chunk_end))

        for start, end in chunks:
            payload = _request_weather_chunk(start, end)
            if payload is None:
                log.warning("Weather fetch failed for %s to %s; leaving as NA", start, end)
                continue
            times = payload.get("time", [])
            for i, day_str in enumerate(times):
                cache[day_str] = {
                    var: payload.get(var, [None] * len(times))[i]
                    for var in WEATHER_DAILY_VARS
                }
            time.sleep(REQUEST_DELAY_SECONDS)

        save_json(WEATHER_CACHE_FILE, cache)

    rows = []
    for d in all_dates:
        record = cache.get(d.isoformat(), {})
        rows.append(
            {
                "Date": pd.Timestamp(d),
                "Temp_Max_C": record.get("temperature_2m_max"),
                "Temp_Min_C": record.get("temperature_2m_min"),
                "Temp_Mean_C": record.get("temperature_2m_mean"),
                "Humidity_Mean_Pct": record.get("relative_humidity_2m_mean"),
                "Precip_Sum_mm": record.get("precipitation_sum"),
                "Rain_Sum_mm": record.get("rain_sum"),
                "Wind_Max_kmh": record.get("wind_speed_10m_max"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Holiday TYPE classification (replaces the old numeric Holiday_cat scale).
#
# Three buckets, in priority order:
#   1. "Eid"        — Eid-ul-Fitr or Eid-ul-Adha (by far the highest-impact
#                      holidays on Dhaka load: multi-day, mass exodus from
#                      the city).
#   2. "Durga Puja"  — the largest Hindu festival in Bangladesh; called out
#                      separately since it has a distinct load signature
#                      (less of a city-exodus effect than Eid, more
#                      localized/commercial impact) from "ordinary" public
#                      holidays.
#   3. "Public Holiday" — every other day the `holidays` library (or the
#                      optional reference CSV) marks as a gazetted holiday:
#                      Victory Day, Independence Day, Pohela Baishakh,
#                      Christmas, Durga Puja's surrounding days, etc.
#   (anything not in the above is "No Holiday".)
# ---------------------------------------------------------------------------
EID_KEYWORDS = {"eid al fitr", "eid ul fitr", "eid al adha", "eid ul adha"}
DURGA_PUJA_KEYWORDS = {"durga puja"}


def classify_holiday_type(holiday_name: str) -> str:
    """Map a (possibly already-standardised) holiday name to one of:
    'Eid', 'Durga Puja', 'Public Holiday'. Caller is responsible for
    handling the 'No Holiday' case separately."""
    clean = normalise_name(holiday_name)
    if any(keyword in clean for keyword in EID_KEYWORDS):
        return "Eid"
    if any(keyword in clean for keyword in DURGA_PUJA_KEYWORDS):
        return "Durga Puja"
    return "Public Holiday"


def holiday_context(dates: pd.Series) -> pd.DataFrame:
    years = sorted(set(pd.to_datetime(dates).dt.year))
    bd_holidays = holidays.country_holidays("BD", years=years)
    reference_holidays = load_reference_holidays()

    rows = []
    for value in pd.to_datetime(dates):
        current_date = value.date()
        is_weekend = _is_weekend(current_date)
        context_extra = {"Is_Weekend": int(is_weekend)}

        # Priority 1: reference CSV
        if current_date in reference_holidays:
            holiday_name = reference_holidays[current_date]
            holiday_type = (
                classify_holiday_type(holiday_name)
                if normalise_name(holiday_name) not in ("", "no holiday")
                else "No Holiday"
            )
            rows.append({"Date": value, "Holiday name": holiday_name,
                         "Holiday_type": holiday_type, **context_extra})
            continue

        # Priority 2: gazette-declared dates (overrides library for extended Eid
        # blocks, and covers Durga Puja which the library omits entirely)
        if current_date in GAZETTE_HOLIDAYS:
            holiday_name, holiday_type = GAZETTE_HOLIDAYS[current_date]
            rows.append({"Date": value, "Holiday name": holiday_name,
                        "Holiday_type": holiday_type, **context_extra})
            continue

        # Priority 3: holidays library (Bengali names → English via BENGALI_TO_ENGLISH)
        raw_holiday = bd_holidays.get(current_date)
        if not raw_holiday:
            rows.append({"Date": value, "Holiday name": "No Holiday",
                         "Holiday_type": "No Holiday", **context_extra})
            continue

        holiday_name = standard_holiday_name(str(raw_holiday))
        holiday_type = classify_holiday_type(holiday_name)
        rows.append({"Date": value, "Holiday name": holiday_name,
                     "Holiday_type": holiday_type, **context_extra})

    return pd.DataFrame(rows)

def make_wide_dataset(df: pd.DataFrame) -> pd.DataFrame:
    load_columns = [f"{substation}_Load_MW" for substation in DHAKA_CITY_SUBSTATIONS]
    time_columns = [f"{substation}_Peak_Time" for substation in DHAKA_CITY_SUBSTATIONS]
    substation_columns = [
        column
        for substation in DHAKA_CITY_SUBSTATIONS
        for column in (f"{substation}_Load_MW", f"{substation}_Peak_Time")
    ]
    if df.empty:
        columns = [
            "Date",
            "Year",
            "Month",
            "Month_Name",
            "DayOfYear",
            "WeekOfYear",
            "Day_of_Week",
            "Season",
            "Season_Encoded",
            "Holiday name",
            "Holiday_type",
            "Is_Weekend",
            "Temp_Max_C",
            "Temp_Min_C",
            "Temp_Mean_C",
            "Humidity_Mean_Pct",
            "Precip_Sum_mm",
            "Rain_Sum_mm",
            "Wind_Max_kmh",
            *substation_columns,
        ]
        return pd.DataFrame(columns=columns)

    load_wide = (
        df.pivot_table(
            index="Date",
            columns="Substation",
            values="Load_MW",
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    load_wide = load_wide.rename(
        columns={
            substation: f"{substation}_Load_MW"
            for substation in DHAKA_CITY_SUBSTATIONS
            if substation in load_wide.columns
        }
    )

    for column in load_columns:
        if column not in load_wide.columns:
            load_wide[column] = pd.NA

    time_wide = (
        df.drop_duplicates(subset=["Date", "Substation"], keep="last")
        .pivot(index="Date", columns="Substation", values="Time")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    time_wide = time_wide.rename(
        columns={
            substation: f"{substation}_Peak_Time"
            for substation in DHAKA_CITY_SUBSTATIONS
            if substation in time_wide.columns
        }
    )
    for column in time_columns:
        if column not in time_wide.columns:
            time_wide[column] = pd.NA

    wide = load_wide[["Date", *load_columns]].merge(
        time_wide[["Date", *time_columns]],
        on="Date",
        how="left",
    )

    out = wide.copy()
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Month_Name"] = out["Date"].dt.strftime("%B")
    out["DayOfYear"] = out["Date"].dt.dayofyear
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype("Int64")
    out["Day_of_Week"] = out["Date"].dt.day_name()
    out["Season"] = out["Month"].map(SEASON_MAP)
    out["Season_Encoded"] = out["Season"].map(SEASON_ENC)
    out = out.merge(holiday_context(out["Date"]), on="Date", how="left")
    out = out.merge(fetch_weather_context(out["Date"]), on="Date", how="left")

    columns = [
        "Date",
        "Year",
        "Month",
        "Month_Name",
        "DayOfYear",
        "WeekOfYear",
        "Day_of_Week",
        "Season",
        "Season_Encoded",
        "Holiday name",
        "Holiday_type",
        "Is_Weekend",
        "Temp_Max_C",
        "Temp_Min_C",
        "Temp_Mean_C",
        "Humidity_Mean_Pct",
        "Precip_Sum_mm",
        "Rain_Sum_mm",
        "Wind_Max_kmh",
        *substation_columns,
    ]
    return out[columns]


def save_outputs(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved CSV: %s (%s rows)", OUTPUT_CSV, len(df))

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Dhaka_City_Substations", index=False)
        summary = pd.DataFrame(
            [
                ("Rows", len(df)),
                ("Substation columns", len(DHAKA_CITY_SUBSTATIONS)),
                ("Date from", df["Date"].min().date().isoformat() if not df.empty else ""),
                ("Date to", df["Date"].max().date().isoformat() if not df.empty else ""),
                (
                    "Mean Load MW across substation columns",
                    round(df[[f"{s}_Load_MW" for s in DHAKA_CITY_SUBSTATIONS]].stack().mean(), 2)
                    if not df.empty
                    else "",
                ),
            ],
            columns=["Metric", "Value"],
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)
    log.info("Saved XLSX: %s", OUTPUT_XLSX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Dhaka city substation CSV from BPDB Page-3 PDFs.")
    parser.add_argument("--start", default=START_DATE.isoformat(), help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE.isoformat(), help="End date, YYYY-MM-DD")
    parser.add_argument(
        "--refresh-recent-days",
        type=int,
        default=REFRESH_RECENT_DAYS,
        help="Redownload and reparse this many latest days on each run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel workers for PDF parsing (default: CPU count minus 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    log.info("BPDB Page-3 Dhaka city substation scraper")
    log.info("Range: %s to %s", start_date, end_date)

    url_cache = build_url_cache(start_date, end_date)
    if not url_cache:
        raise SystemExit("No Page-3 URLs found. Check network access or BPDB archive availability.")

    download_missing_pdfs(
        url_cache=url_cache,
        start_date=start_date,
        end_date=end_date,
        refresh_recent_days=args.refresh_recent_days,
    )

    raw_df = parse_all_pdfs(
        url_cache=url_cache,
        start_date=start_date,
        end_date=end_date,
        refresh_recent_days=args.refresh_recent_days,
        workers=args.workers,
    )
    final_df = make_wide_dataset(raw_df)
    save_outputs(final_df)

    if final_df.empty:
        log.warning("Output is empty. Usually this means PDFs were not downloaded or table parsing needs tuning.")
    else:
        log.info(
            "Done: %s rows, %s substations, %s to %s",
            len(final_df),
            len(DHAKA_CITY_SUBSTATIONS),
            final_df["Date"].min().date(),
            final_df["Date"].max().date(),
        )


if __name__ == "__main__":
    main()