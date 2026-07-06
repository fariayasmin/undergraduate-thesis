# Using this knowledge graph with an LLM (GraphRAG)

Purpose: let an LLM answer customer questions such as
"Why was my AC rescheduled on 2025-05-14?" or
"How much did household DHH0042 save last month, and at what comfort cost?"

## Retrieval pattern
1. Resolve the entity (household / appliance / date) from the question.
2. Cypher-retrieve the local subgraph, e.g.:
   MATCH (u:User {household_id:$hh})-[:EXPERIENCED]->(e:DrEvent)
   WHERE e.date = $date
   OPTIONAL MATCH (u)-[:HAS_METER]->(m)-[:REGISTERS]->(a:Appliance)
            -[:BELONGS_TO]->(c:Category)
   RETURN u, e, collect({appliance:a.appliance_name, category:c.name,
                         template:c.explanation_template})
3. Serialise the returned nodes/edges as short triples
   (User KHH0007) -IN_TIER-> (middle, 35k-65k BDT)
   (DrEvent DR_KAL_00012) date=2025-05-14 shifted=1.42kWh curtailed=0.36kWh
4. Feed triples + the Category.explanation_template into the LLM prompt and
   ask it to compose the answer. DrEvent.explanation already contains a
   rendered fallback sentence, so even a template-only system is grounded.

## Why the graph grounds the LLM
- Every DR action links to the exact MOLP inputs: tier (alpha weight),
  category constraint (must_run / flexible_start / power_reduction_allowed),
  TOU period and tariff slab -- the LLM never has to invent numbers.
- flexibility_index on BELONGS_TO lets the LLM rank which appliance was
  chosen first and say so ("your washing machine is 14x more flexible
  than your refrigerator").
- Substation nodes carry the real BPDB yearly peak means, so grid-level
  answers ("why was 2025-05-14 a stress day?") tie back to observed data.
- MonthlyBill nodes ((:User)-[:BILLED {month}]->(:MonthlyBill)) hold each
  household's baseline vs optimised bill, saving and discomfort per month,
  so "how much did DHH0042 save last month, and at what comfort cost?" is a
  single-hop lookup:
    MATCH (u:User {household_id:$hh})-[:BILLED]->(b:MonthlyBill {month:$ym})
    RETURN b.baseline_bill_bdt, b.optimised_bill_bdt, b.saving_bdt,
           b.discomfort_bdt_eq
