// ============================================================================
// load_kg_full.cypher — Kalyanpur+Dhanmondi DR knowledge graph
// SAFE TO RE-RUN: uses MERGE, so running it twice never duplicates anything.
// Put all 21 CSVs in the Neo4j /import folder before running.
//
// Paste the whole file into Neo4j Browser (it runs statement by statement),
// or run:  cat load_kg_full.cypher | cypher-shell -u neo4j -p <password>
// ============================================================================

// ── 0. Clean slate ──────────────────────────────────────────────────────────
// Deletes ALL existing nodes and relationships (batched, safe on small heaps).
CALL {
  MATCH (n) DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;

// The ORIGINAL script created one constraint on the generic :Node label.
// It has an auto-generated name, so it can't be dropped here by a fixed name.
// Run  SHOW CONSTRAINTS;  — if you see a row with labelsOrTypes = ["Node"],
// drop it once by its name:   DROP CONSTRAINT <that_name>;
// (Harmless if left in place, but it clutters the schema.)

// ── 1. Constraints (one per label; IF NOT EXISTS makes re-runs safe) ────────
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Substation)        REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Ward)              REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:User)              REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Meter)             REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Appliance)         REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Category)          REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:SocioeconomicTier) REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:TariffSlab)        REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:TouPeriod)         REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:DrEvent)           REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:MonthlyBill)       REQUIRE n.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:ForecastDay)       REQUIRE n.nodeId IS UNIQUE;

// ── 2. Nodes (MERGE on nodeId, then SET all properties) ─────────────────────
LOAD CSV WITH HEADERS FROM 'file:///nodes_Substation.csv' AS r
MERGE (n:Substation {nodeId: r.nodeId})
SET n.name = r.name, n.city = r.city, n.division = r.division,
    n.utility = r.utility,
    n.mean_daily_peak_mw_2024 = toFloat(r.mean_daily_peak_mw_2024),
    n.mean_daily_peak_mw_2025 = toFloat(r.mean_daily_peak_mw_2025),
    n.mean_daily_peak_mw_2026 = toFloat(r.mean_daily_peak_mw_2026),
    n.feeder_voltage_kv = toFloat(r.feeder_voltage_kv),
    n.latitude = toFloat(r.latitude), n.longitude = toFloat(r.longitude);

LOAD CSV WITH HEADERS FROM 'file:///nodes_Ward.csv' AS r
MERGE (n:Ward {nodeId: r.nodeId})
SET n.name = r.name, n.substation_id = r.substation_id;

LOAD CSV WITH HEADERS FROM 'file:///nodes_User.csv' AS r
MERGE (n:User {nodeId: r.nodeId})
SET n.household_id = r.household_id, n.tier = r.tier,
    n.household_size = toInteger(r.household_size),
    n.monthly_income_bdt = toInteger(r.monthly_income_bdt),
    n.avg_monthly_bill_bdt = toFloat(r.avg_monthly_bill_bdt),
    n.substation = r.substation;

LOAD CSV WITH HEADERS FROM 'file:///nodes_Meter.csv' AS r
MERGE (n:Meter {nodeId: r.nodeId})
SET n.meter_id = r.meter_id, n.household_id = r.household_id,
    n.meter_type = r.meter_type, n.substation_id = r.substation_id,
    n.ward_id = r.ward_id, n.monthly_kwh_est = toFloat(r.monthly_kwh_est);

LOAD CSV WITH HEADERS FROM 'file:///nodes_Appliance.csv' AS r
MERGE (n:Appliance {nodeId: r.nodeId})
SET n.appliance_id = r.appliance_id, n.appliance_name = r.appliance_name,
    n.category = r.category,
    n.rated_power_w = toInteger(r.rated_power_w),
    n.actual_power_w = toFloat(r.actual_power_w),
    n.duty_cycle = toFloat(r.duty_cycle),
    n.quantity = toInteger(r.quantity),
    n.household_id = r.household_id;

LOAD CSV WITH HEADERS FROM 'file:///nodes_Category.csv' AS r
MERGE (n:Category {nodeId: r.nodeId})
SET n.name = r.name,
    n.curtailment_priority = toInteger(r.curtailment_priority),
    n.can_curtail = toBoolean(r.can_curtail),
    n.can_schedule = toBoolean(r.can_schedule),
    n.molp_constraint = r.molp_constraint,
    n.explanation_template = r.explanation_template;

LOAD CSV WITH HEADERS FROM 'file:///nodes_SocioeconomicTier.csv' AS r
MERGE (n:SocioeconomicTier {nodeId: r.nodeId})
SET n.name = r.name,
    n.income_min_bdt = toInteger(r.income_min_bdt),
    n.income_max_bdt = toInteger(r.income_max_bdt);

LOAD CSV WITH HEADERS FROM 'file:///nodes_TariffSlab.csv' AS r
MERGE (n:TariffSlab {nodeId: r.nodeId})
SET n.slab_id = toInteger(r.slab_id),
    n.from_kwh = toInteger(r.from_kwh), n.to_kwh = toInteger(r.to_kwh),
    n.rate_bdt_kwh = toFloat(r.rate_bdt_kwh),
    n.tou_peak_rate = toFloat(r.tou_peak_rate),
    n.tou_offpeak_rate = toFloat(r.tou_offpeak_rate);

LOAD CSV WITH HEADERS FROM 'file:///nodes_TouPeriod.csv' AS r
MERGE (n:TouPeriod {nodeId: r.nodeId})
SET n.name = r.name, n.hours = r.hours,
    n.multiplier = toFloat(r.multiplier);

LOAD CSV WITH HEADERS FROM 'file:///nodes_DrEvent.csv' AS r
MERGE (n:DrEvent {nodeId: r.nodeId})
SET n.date = date(r.date),                      // native Date (YYYY-MM-DD)
    n.substation = r.substation, n.action = r.action,
    // day-ahead substation forecast (selected model, 05b) that declared
    // the DR window — grounds "why was this a stress day?" answers
    n.substation_pred_peak_mw = toFloat(r.substation_pred_peak_mw),
    n.temp_max_c = toFloat(r.temp_max_c),      // that day's max temperature
    // GCS decomposition inputs (GCS = 0.4*PDR + 0.3*PMN + 0.3*AA)
    n.price_multiplier_norm = toFloat(r.price_multiplier_norm),
    n.appliance_alpha = toFloat(r.appliance_alpha),
    n.energy_shifted_kwh = toFloat(r.energy_shifted_kwh),
    n.energy_curtailed_kwh = toFloat(r.energy_curtailed_kwh),
    n.molp_alpha = toFloat(r.molp_alpha),
    n.gcs_score = toFloat(r.gcs_score),
    n.predicted_load_w = toFloat(r.predicted_load_w),
    n.optimised_load_w = toFloat(r.optimised_load_w),
    n.power_deviation_ratio = toFloat(r.power_deviation_ratio),
    n.explanation = r.explanation;

LOAD CSV WITH HEADERS FROM 'file:///nodes_MonthlyBill.csv' AS r
MERGE (n:MonthlyBill {nodeId: r.nodeId})
SET n.household_id = r.household_id, n.month = r.month,
    n.baseline_bill_bdt = toFloat(r.baseline_bill_bdt),
    n.optimised_bill_bdt = toFloat(r.optimised_bill_bdt),
    n.saving_bdt = toFloat(r.saving_bdt),
    n.saving_pct = toFloat(r.saving_pct),
    n.energy_kwh = toFloat(r.energy_kwh),
    n.energy_shifted_kwh = toFloat(r.energy_shifted_kwh),
    n.energy_curtailed_kwh = toFloat(r.energy_curtailed_kwh),
    n.discomfort_bdt_eq = toFloat(r.discomfort_bdt_eq),
    // month weather context: grounds "it was hot/humid -> AC ran more"
    n.mean_temp_c = toFloat(r.mean_temp_c),
    n.max_temp_c = toFloat(r.max_temp_c),
    n.hot_days = toInteger(r.hot_days),
    n.mean_humidity_pct = toFloat(r.mean_humidity_pct),
    n.rain_mm = toFloat(r.rain_mm);

LOAD CSV WITH HEADERS FROM 'file:///nodes_ForecastDay.csv' AS r
MERGE (n:ForecastDay {nodeId: r.nodeId})
SET n.date = date(r.date), n.substation = r.substation,
    n.pred_peak_mw = toFloat(r.pred_peak_mw),
    n.stress_threshold_mw = toFloat(r.stress_threshold_mw),
    n.is_stress_day = toBoolean(r.is_stress_day),
    n.temp_max_c = toFloat(r.temp_max_c);

// ── 3. Relationships (MERGE = no duplicate edges on re-run) ─────────────────
LOAD CSV WITH HEADERS FROM 'file:///rels_HAS_METER.csv' AS r
MATCH (a:User  {nodeId: r.`:START_ID`})
MATCH (b:Meter {nodeId: r.`:END_ID`})
MERGE (a)-[:HAS_METER]->(b);

LOAD CSV WITH HEADERS FROM 'file:///rels_FEEDS_FROM.csv' AS r
MATCH (a:Meter      {nodeId: r.`:START_ID`})
MATCH (b:Substation {nodeId: r.`:END_ID`})
MERGE (a)-[:FEEDS_FROM]->(b);

LOAD CSV WITH HEADERS FROM 'file:///rels_LOCATED_IN.csv' AS r
MATCH (a:Meter {nodeId: r.`:START_ID`})
MATCH (b:Ward  {nodeId: r.`:END_ID`})
MERGE (a)-[:LOCATED_IN]->(b);

LOAD CSV WITH HEADERS FROM 'file:///rels_PART_OF.csv' AS r
MATCH (a:Ward       {nodeId: r.`:START_ID`})
MATCH (b:Substation {nodeId: r.`:END_ID`})
MERGE (a)-[:PART_OF]->(b);

LOAD CSV WITH HEADERS FROM 'file:///rels_IN_TIER.csv' AS r
MATCH (a:User              {nodeId: r.`:START_ID`})
MATCH (b:SocioeconomicTier {nodeId: r.`:END_ID`})
MERGE (a)-[rel:IN_TIER]->(b)
SET rel.income_bdt = toInteger(r.income_bdt);

LOAD CSV WITH HEADERS FROM 'file:///rels_REGISTERS.csv' AS r
MATCH (a:Meter     {nodeId: r.`:START_ID`})
MATCH (b:Appliance {nodeId: r.`:END_ID`})
MERGE (a)-[:REGISTERS]->(b);

LOAD CSV WITH HEADERS FROM 'file:///rels_INSTALLED_AT.csv' AS r
MATCH (a:Appliance {nodeId: r.`:START_ID`})
MATCH (b:Meter     {nodeId: r.`:END_ID`})
MERGE (a)-[:INSTALLED_AT]->(b);

LOAD CSV WITH HEADERS FROM 'file:///rels_BELONGS_TO.csv' AS r
MATCH (a:Appliance {nodeId: r.`:START_ID`})
MATCH (b:Category  {nodeId: r.`:END_ID`})
MERGE (a)-[rel:BELONGS_TO]->(b)
SET rel.flexibility_index = toFloat(r.flexibility_index);

LOAD CSV WITH HEADERS FROM 'file:///rels_SUBJECT_TO.csv' AS r
MATCH (a:User       {nodeId: r.`:START_ID`})
MATCH (b:TariffSlab {nodeId: r.`:END_ID`})
MERGE (a)-[rel:SUBJECT_TO]->(b)
SET rel.monthly_kwh_est = toFloat(r.monthly_kwh_est);

// rels_EXPERIENCED.csv date format changed across pipeline versions:
// legacy exports used DD-MM-YY (29-04-24); regenerated exports use ISO
// YYYY-MM-DD (2024-04-29). Detect by the first segment's length and
// normalise both to a native Date.
LOAD CSV WITH HEADERS FROM 'file:///rels_EXPERIENCED.csv' AS r
MATCH (a:User    {nodeId: r.`:START_ID`})
MATCH (b:DrEvent {nodeId: r.`:END_ID`})
WITH a, b, r.date AS ds, split(r.date, '-') AS p
MERGE (a)-[rel:EXPERIENCED]->(b)
SET rel.date = CASE WHEN size(p[0]) = 4
                    THEN date(left(ds, 10))                       // ISO
                    ELSE date('20' + p[2] + '-' + p[1] + '-' + p[0])  // DD-MM-YY
               END;

LOAD CSV WITH HEADERS FROM 'file:///rels_DURING.csv' AS r
MATCH (a:DrEvent   {nodeId: r.`:START_ID`})
MATCH (b:TouPeriod {nodeId: r.`:END_ID`})
MERGE (a)-[:DURING]->(b);

LOAD CSV WITH HEADERS FROM 'file:///rels_BILLED.csv' AS r
MATCH (a:User        {nodeId: r.`:START_ID`})
MATCH (b:MonthlyBill {nodeId: r.`:END_ID`})
MERGE (a)-[rel:BILLED]->(b)
SET rel.month = r.month;

LOAD CSV WITH HEADERS FROM 'file:///rels_FORECAST_FOR.csv' AS r
MATCH (a:ForecastDay {nodeId: r.`:START_ID`})
MATCH (b:Substation  {nodeId: r.`:END_ID`})
MERGE (a)-[:FORECAST_FOR]->(b);

// ── 4. Verify ───────────────────────────────────────────────────────────────
// Expected node counts:
//   Appliance 2973, DrEvent 800, User 240, Meter 240, Ward 10, TariffSlab 7,
//   Category 3, SocioeconomicTier 3, TouPeriod 3, Substation 2
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS nodes ORDER BY nodes DESC;

// Expected relationship counts:
//   REGISTERS 2973, INSTALLED_AT 2973, BELONGS_TO 2973,
//   EXPERIENCED 800, DURING 800,
//   HAS_METER 240, FEEDS_FROM 240, LOCATED_IN 240, IN_TIER 240, SUBJECT_TO 240,
//   PART_OF 10
MATCH ()-[rel]->() RETURN type(rel) AS relationship, count(*) AS edges ORDER BY edges DESC;
