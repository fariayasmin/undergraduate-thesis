// Load the Kalyanpur+Dhanmondi DR knowledge graph (put CSVs in /import)
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.nodeId IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///nodes_Substation.csv' AS r
  CREATE (:Substation:Node {nodeId:r.nodeId, name:r.name, utility:r.utility,
    latitude:toFloat(r.latitude), longitude:toFloat(r.longitude)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_Ward.csv' AS r
  CREATE (:Ward:Node {nodeId:r.nodeId, name:r.name});
LOAD CSV WITH HEADERS FROM 'file:///nodes_User.csv' AS r
  CREATE (:User:Node {nodeId:r.nodeId, household_id:r.household_id, tier:r.tier,
    monthly_income_bdt:toInteger(r.monthly_income_bdt), substation:r.substation});
LOAD CSV WITH HEADERS FROM 'file:///nodes_Meter.csv' AS r
  CREATE (:Meter:Node {nodeId:r.nodeId, meter_id:r.meter_id,
    monthly_kwh_est:toFloat(r.monthly_kwh_est)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_Appliance.csv' AS r
  CREATE (:Appliance:Node {nodeId:r.nodeId, appliance_name:r.appliance_name,
    category:r.category, rated_power_w:toInteger(r.rated_power_w)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_Category.csv' AS r
  CREATE (:Category:Node {nodeId:r.nodeId, name:r.name,
    explanation_template:r.explanation_template, molp_constraint:r.molp_constraint});
LOAD CSV WITH HEADERS FROM 'file:///nodes_SocioeconomicTier.csv' AS r
  CREATE (:SocioeconomicTier:Node {nodeId:r.nodeId, name:r.name});
LOAD CSV WITH HEADERS FROM 'file:///nodes_TariffSlab.csv' AS r
  CREATE (:TariffSlab:Node {nodeId:r.nodeId, slab_id:toInteger(r.slab_id),
    rate_bdt_kwh:toFloat(r.rate_bdt_kwh)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_TouPeriod.csv' AS r
  CREATE (:TouPeriod:Node {nodeId:r.nodeId, name:r.name,
    multiplier:toFloat(r.multiplier)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_DrEvent.csv' AS r
  CREATE (:DrEvent:Node {nodeId:r.nodeId, date:r.date, action:r.action,
    substation_pred_peak_mw:toFloat(r.substation_pred_peak_mw),
    temp_max_c:toFloat(r.temp_max_c),
    power_deviation_ratio:toFloat(r.power_deviation_ratio),
    price_multiplier_norm:toFloat(r.price_multiplier_norm),
    appliance_alpha:toFloat(r.appliance_alpha),
    gcs_score:toFloat(r.gcs_score),
    actual_substation_peak_mw:toFloat(r.actual_substation_peak_mw),
    forecast_error_mw:toFloat(r.forecast_error_mw),
    stress_threshold_mw:toFloat(r.stress_threshold_mw),
    rebate_bdt:toFloat(r.rebate_bdt),
    explanation:r.explanation});
LOAD CSV WITH HEADERS FROM 'file:///nodes_ForecastDay.csv' AS r
  CREATE (:ForecastDay:Node {nodeId:r.nodeId, date:r.date,
    substation:r.substation, pred_peak_mw:toFloat(r.pred_peak_mw),
    actual_peak_mw:toFloat(r.actual_peak_mw),
    forecast_error_mw:toFloat(r.forecast_error_mw),
    stress_threshold_mw:toFloat(r.stress_threshold_mw),
    is_stress_day:toBoolean(r.is_stress_day),
    temp_max_c:toFloat(r.temp_max_c)});
LOAD CSV WITH HEADERS FROM 'file:///nodes_MonthlyBill.csv' AS r
  CREATE (:MonthlyBill:Node {nodeId:r.nodeId, month:r.month,
    baseline_bill_bdt:toFloat(r.baseline_bill_bdt),
    optimised_bill_bdt:toFloat(r.optimised_bill_bdt),
    saving_bdt:toFloat(r.saving_bdt), saving_pct:toFloat(r.saving_pct),
    energy_kwh:toFloat(r.energy_kwh),
    mean_temp_c:toFloat(r.mean_temp_c), max_temp_c:toFloat(r.max_temp_c),
    hot_days:toInteger(r.hot_days),
    mean_humidity_pct:toFloat(r.mean_humidity_pct),
    rain_mm:toFloat(r.rain_mm),
    discomfort_bdt_eq:toFloat(r.discomfort_bdt_eq)});
// Relationships (repeat pattern for every rels_*.csv)
LOAD CSV WITH HEADERS FROM 'file:///rels_HAS_METER.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:HAS_METER]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_FEEDS_FROM.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:FEEDS_FROM]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_LOCATED_IN.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:LOCATED_IN]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_PART_OF.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:PART_OF]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_IN_TIER.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:IN_TIER {income_bdt:toInteger(r.income_bdt)}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_REGISTERS.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:REGISTERS]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_INSTALLED_AT.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:INSTALLED_AT]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_BELONGS_TO.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:BELONGS_TO {flexibility_index:toFloat(r.flexibility_index)}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_SUBJECT_TO.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:SUBJECT_TO {monthly_kwh_est:toFloat(r.monthly_kwh_est)}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_EXPERIENCED.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:EXPERIENCED {date:r.date}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_DURING.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:DURING]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_BILLED.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:BILLED {month:r.month}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///rels_FORECAST_FOR.csv' AS r
  MATCH (a:Node {nodeId:r.`:START_ID`}), (b:Node {nodeId:r.`:END_ID`})
  CREATE (a)-[:FORECAST_FOR]->(b);
