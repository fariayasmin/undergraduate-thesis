// GenTwin-SG :: Cypher query pack  (schema v2, billing-enabled)
//
// Graph shape, after the reviewer's structural corrections:
//
//   (Substation)-[:HAS_EVENT]->(Event)-[:AFFECTS]->(Household)
//   (Event)-[:CAUSES]->(DrAction)-[:AT_TIMESLOT]->(TimeSlot)
//   (Event)-[:TRIGGERED_BY]->(Forecast)      (Event)-[:UNDER_REGIME]->(Regime)
//   (Household)-[:HAS_APPLIANCE]->(Appliance)-[:BELONGS_TO]->(ApplianceClass)
//   (Household)-[:BILLED]->(MonthlyBill)-[:BILLED_UNDER]->(BillingPeriod)
//
// NOTE ON SCALE: a :Household node is a REPRESENTATIVE consumer standing for
// h.w_h real connections. Population figures must be weighted by w_h; a bare
// count(h) counts agents, not customers. Every query below that reports a
// population quantity does this.

// ---------------------------------------------------------------------------
// Q1  "Why was Dhanmondi under demand response on 2026-07-01?"
//     Forecast -> Regime -> Event -> Household, with the stored Prop. 2 terms.
// ---------------------------------------------------------------------------
MATCH (s:Substation {name:'Dhanmondi'})-[:HAS_EVENT]->(e:Event {date:'2026-07-01'})
MATCH (e)-[:TRIGGERED_BY]->(f:Forecast)
MATCH (e)-[:UNDER_REGIME]->(r:Regime)
OPTIONAL MATCH (e)-[:AFFECTS]->(h:Household)
RETURN e.trigger, r.r AS regime, r.regime_name, r.s_stress,
       f.p_hat_kw, f.p_tilde_kw, f.sigma_kw, f.z_beta, f.p_str_kw, f.kappa,
       count(DISTINCT h) AS representative_agents,
       sum(DISTINCT h.w_h)  AS real_connections_affected;

// ---------------------------------------------------------------------------
// Q2  All emergency events and the customers they affected
// ---------------------------------------------------------------------------
MATCH (e:Event)-[:UNDER_REGIME]->(r:Regime)
WHERE e.regime = 3
OPTIONAL MATCH (e)-[:AFFECTS]->(h:Household)
RETURN e.event_id, e.substation, e.date, e.trigger,
       r.trigger_transfer, r.trigger_reserve_shortfall, r.trigger_capacity_deficit,
       count(DISTINCT h) AS agents, sum(DISTINCT h.w_h) AS connections,
       e.energy_curtailed_kwh, e.energy_shifted_kwh
ORDER BY connections DESC;

// ---------------------------------------------------------------------------
// Q3  Which households provided the most flexibility?  (w_h-weighted)
// ---------------------------------------------------------------------------
MATCH (h:Household)-[:PERFORMED]->(a:DrAction)
MATCH (h)-[:HAS_APPLIANCE]->(ap:Appliance)-[:BELONGS_TO]->(c:ApplianceClass)
WHERE c.name IN ['shiftable','curtailable']
RETURN h.household_id, h.category, h.archetype, h.income_band,
       h.w_h AS represents_connections,
       h.lambda_h, h.gamma_cu_h, h.y_max_h,
       sum(a.energy_kwh)         AS kwh_per_agent,
       sum(a.energy_kwh) * h.w_h AS kwh_population,
       collect(DISTINCT ap.name)[0..6] AS flexible_appliances
ORDER BY kwh_population DESC LIMIT 20;

// ---------------------------------------------------------------------------
// Q4  Which households curtailed at 18:30?   (slot level)
// ---------------------------------------------------------------------------
MATCH (h:Household)-[:PERFORMED]->(a:DrAction {action:'curtail', clock:'18:30'})
      -[:AT_TIMESLOT]->(t:TimeSlot)
RETURN t.date, h.household_id, h.category, h.w_h,
       a.value AS y_star, a.energy_kwh,
       a.private_tk_per_kwh, a.system_tk_per_kwh, a.qoe_tk_per_kwh,
       a.discomfort_tk_per_kwh, a.margin, a.threshold_form, a.c7_binding
ORDER BY a.energy_kwh DESC LIMIT 25;

// ---------------------------------------------------------------------------
// Q5  Battery state and shadow price at 17:00
// ---------------------------------------------------------------------------
MATCH (s:Substation)-[:HAS_PROFILE]->(l:LoadProfile {clock:'17:00'})
RETURN l.date, s.name, l.soc_kwh, l.pi_tk_per_kwh, l.post_response_kw,
       l.c1_slack_kw ORDER BY l.date, s.name;

// ---------------------------------------------------------------------------
// Q6  Commercial participation between 16:00 and 18:00
// ---------------------------------------------------------------------------
MATCH (h:Household {category:'Commercial'})-[:PERFORMED]->(a:DrAction)
      -[:AT_TIMESLOT]->(t:TimeSlot)
WHERE t.slot_index >= 32 AND t.slot_index < 36
RETURN t.date, t.clock, count(DISTINCT h) AS agents,
       sum(DISTINCT h.w_h) AS connections, sum(a.energy_kwh) AS kwh
ORDER BY t.date, t.clock;

// ---------------------------------------------------------------------------
// Q7  Which hospital loads were never curtailed?   (absence, one hop)
//     Answer should be ALL of them - and by threshold, not by rule.
// ---------------------------------------------------------------------------
MATCH (p:ParticipationSummary {never_curtailed:true})-[:SUMMARISES]->(h:Household)
WHERE h.category = 'Hospital'
RETURN p.date, count(h) AS agents, sum(h.w_h) AS connections,
       avg(h.lambda_h) AS mean_lambda, avg(h.gamma_cu_h) AS mean_gamma_cu,
       collect(h.household_id)[0..8] AS examples;

// ---------------------------------------------------------------------------
// Q8  Why did this action happen?   The stored threshold, no solver re-run.
//     LHS = private + system ; RHS = discomfort + QoE dual (when C7 binds)
// ---------------------------------------------------------------------------
MATCH (h:Household)-[:PERFORMED]->(a:DrAction)
WHERE a.margin IS NOT NULL
RETURN h.household_id, a.date, a.clock, a.action, a.value,
       a.private_tk_per_kwh + a.system_tk_per_kwh          AS lhs,
       a.discomfort_tk_per_kwh + a.qoe_tk_per_kwh          AS rhs,
       a.margin, a.lambda_h, a.lambda_star, a.pi_i_t, a.kappa,
       a.c7_binding, a.at_cap, a.equation
ORDER BY a.margin DESC LIMIT 20;

// ---------------------------------------------------------------------------
// Q9  Which optimisation decisions caused an event?
// ---------------------------------------------------------------------------
MATCH (e:Event)-[:COMPUTED_FROM]->(run:OptimizationRun)-[:PRODUCED]->(d:Decision)
WHERE d.substation = e.substation AND d.value > 0
RETURN e.event_id, d.var_type, count(*) AS slots,
       sum(d.value) AS total, max(d.value) AS peak_value
ORDER BY e.event_id, total DESC;

// ---------------------------------------------------------------------------
// Q10 Full LLM context bundle for one household-day, one round trip
// ---------------------------------------------------------------------------
MATCH (h:Household {household_id:'DHNC0001'})
OPTIONAL MATCH (h)-[:IN_ARCHETYPE]->(ar:Archetype)-[:SUBJECT_TO]->(tc:TariffClass)
OPTIONAL MATCH (h)-[:LOCATED_AT]->(s:Substation)-[:HAS_TOU_WINDOW]->(tw:TouWindow)
OPTIONAL MATCH (h)-[:HAS_APPLIANCE]->(ap:Appliance)
OPTIONAL MATCH (h)-[:PERFORMED]->(a:DrAction {date:'2026-07-01'})
OPTIONAL MATCH (ps:ParticipationSummary {date:'2026-07-01'})-[:SUMMARISES]->(h)
OPTIONAL MATCH (h)-[:BILLED]->(b:MonthlyBill)
OPTIONAL MATCH (f:Forecast {substation:'Dhanmondi', date:'2026-07-01'})
OPTIONAL MATCH (rg:Regime {substation:'Dhanmondi', date:'2026-07-01'})
RETURN h{.household_id,.category,.archetype,.income_band,.household_size,
         .w_h,.lambda_h,.gamma_cu_h,.y_max_h}                     AS household,
       tc{.code,.name_en,.flat_tk_per_kwh}                        AS tariff,
       tw{.window_label,.mu_peak,.mu_off,.mu_source,.window_source} AS tou,
       f{.p_hat_kw,.p_tilde_kw,.sigma_kw,.kappa,.s_stress,.p_str_kw} AS forecast,
       rg{.r,.regime_name,.trigger_transfer,.trigger_reserve_shortfall} AS regime,
       b{.energy_before_kwh,.energy_after_kwh,.bill_before_tk,.bill_after_tk,
         .bill_saving_tk,.incentive_received_tk,.discomfort_cost_tk,
         .net_benefit_tk,.gained,.qoe,.n_dr_events}               AS monthly_bill,
       ps{.energy_curtailed_kwh,.energy_shifted_kwh,.never_curtailed} AS day,
       collect(DISTINCT ap{.name,.cls,.p_rat_kw,.quantity,.daily_hours})[0..10]
                                                                  AS appliances,
       collect(DISTINCT a{.clock,.action,.value,.energy_kwh,.private_tk_per_kwh,
                          .system_tk_per_kwh,.discomfort_tk_per_kwh,
                          .margin})[0..12]                        AS actions;

// ---------------------------------------------------------------------------
// Q11 Priority pool, ranked
// ---------------------------------------------------------------------------
MATCH (p:PriorityPool)-[c:CONTAINS]->(pe:PoolEntity)
RETURN p.date, p.pool, p.rule, p.degenerate, p.W_P, p.chi_bar, p.alpha_P,
       c.rank, pe.entity_id, pe.kind, pe.chi
ORDER BY p.date, c.rank LIMIT 40;

// ---------------------------------------------------------------------------
// Q12 "Why is my monthly bill what it is, and did demand response help me?"
//     Scenario A (no DR) vs Scenario B (DR), on one node.
// ---------------------------------------------------------------------------
MATCH (h:Household {household_id:'DHNC0001'})-[:BILLED]->(b:MonthlyBill)
      -[:BILLED_UNDER]->(p:BillingPeriod)
OPTIONAL MATCH (b)-[:SUBJECT_TO]->(tc:TariffClass)
RETURN h.household_id, h.income_band, h.household_size,
       h.w_h AS represents_connections,
       p.start, p.end, p.days, p.observed_days, p.forecast_days,
       tc.code, b.slab_before, b.slab_after,
       b.energy_before_kwh, b.energy_after_kwh, b.energy_saving_kwh,
       b.bill_before_tk, b.bill_after_tk, b.bill_saving_tk, b.bill_saving_pct,
       b.incentive_received_tk, b.discomfort_cost_tk,
       b.net_benefit_tk, b.gained, b.qoe, b.n_dr_events, b.peak_reduction_pct,
       b.J1_before_tk, b.J1_after_tk, b.marginal_rate_tk_per_kwh;

// ---------------------------------------------------------------------------
// Q13 Who gained and who lost, by income band   (w_h-weighted)
// ---------------------------------------------------------------------------
MATCH (h:Household)-[:BILLED]->(b:MonthlyBill)
RETURN h.substation, h.income_band,
       count(*)          AS agents,
       sum(b.w_h)        AS connections,
       sum(CASE WHEN b.gained THEN b.w_h ELSE 0 END) AS connections_gained,
       round(100.0 * sum(CASE WHEN b.gained THEN b.w_h ELSE 0 END)
             / sum(b.w_h)) AS pct_gained,
       round(sum(b.bill_saving_tk      * b.w_h)) AS total_saving_tk,
       round(sum(b.incentive_received_tk * b.w_h)) AS total_incentive_tk,
       round(sum(b.discomfort_cost_tk  * b.w_h)) AS total_discomfort_tk,
       round(sum(b.net_benefit_tk      * b.w_h)) AS total_net_benefit_tk,
       round(avg(b.qoe)*1000)/1000 AS mean_qoe
ORDER BY h.substation, h.income_band;

// ---------------------------------------------------------------------------
// Q14 Trace a bill saving back to the actions that produced it
// ---------------------------------------------------------------------------
MATCH (h:Household)-[:BILLED]->(b:MonthlyBill)
WHERE b.bill_saving_tk > 0
MATCH (h)-[:PERFORMED]->(a:DrAction)
RETURN h.household_id, h.category, b.w_h,
       b.bill_saving_tk, b.incentive_received_tk, b.discomfort_cost_tk,
       b.net_benefit_tk,
       count(a) AS actions, sum(a.energy_kwh) AS kwh,
       collect(DISTINCT a.clock)[0..8] AS hours,
       avg(a.private_tk_per_kwh)    AS mean_private,
       avg(a.system_tk_per_kwh)     AS mean_system,
       avg(a.discomfort_tk_per_kwh) AS mean_discomfort
ORDER BY b.bill_saving_tk DESC LIMIT 15;

// ---------------------------------------------------------------------------
// Q15 Event -> affected customers -> the actions it caused   (full chain)
// ---------------------------------------------------------------------------
MATCH (s:Substation)-[:HAS_EVENT]->(e:Event {trigger:'demand_response'})
MATCH (e)-[:AFFECTS]->(h:Household)
MATCH (e)-[:CAUSES]->(a:DrAction)
WHERE a.consumer_id = h.household_id
RETURN e.event_id, e.date, s.name,
       count(DISTINCT h) AS agents, sum(DISTINCT h.w_h) AS connections,
       count(a) AS actions, sum(a.energy_kwh) AS kwh,
       collect(DISTINCT h.category) AS categories
ORDER BY kwh DESC;

// ---------------------------------------------------------------------------
// Q16 Population sanity: agents vs real connections
//     Guards against reporting agent counts as customer counts.
// ---------------------------------------------------------------------------
MATCH (h:Household)
RETURN h.substation, h.category,
       count(*)      AS representative_agents,
       round(sum(h.w_h)) AS real_connections,
       round(avg(h.w_h)) AS mean_agents_weight;

// ---------------------------------------------------------------------------
// Sanity: the causal chain is complete for every event
// ---------------------------------------------------------------------------
MATCH (e:Event)
RETURN e.event_id,
       exists((e)-[:TRIGGERED_BY]->(:Forecast))         AS has_forecast,
       exists((e)-[:UNDER_REGIME]->(:Regime))           AS has_regime,
       exists((e)-[:COMPUTED_FROM]->(:OptimizationRun)) AS has_run,
       exists((e)-[:AFFECTS]->(:Household))             AS has_affected;
