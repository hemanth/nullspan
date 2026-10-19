import test from 'node:test';
import assert from 'node:assert/strict';
import extract from '../src/index.js';

test('extracts, computes, and abstains on support claim email', () => {
  const email =
    "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. " +
    "I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this fixed or refunded. Thanks, Dana";

  const { answers } = extract(email, {
    customer_name: { type: 'text', instructions: "Customer's first name" },
    order_id: { type: 'text', instructions: 'Order number' },
    delivered_on: { type: 'date', instructions: 'Delivery date' },
    amount_paid: { type: 'number', decimals: 2, unit: 'USD', instructions: 'Total paid including tax' },
    days_since_delivery: { type: 'duration', unit: 'days', integer: true },
    damage: { type: 'list', items: 'text', instructions: 'Each kind of damage described' },
    tracking_number: { type: 'text' },
    repair_cost: { type: 'estimate', unit: 'USD', min: 10, max: 5000, scale: 'log', step: 1 },
  });

  assert.equal(answers.customer_name.value, 'Dana');
  assert.equal(answers.order_id.value, 'A-88412');
  assert.deepEqual(answers.order_id.span, [10, 17]);
  assert.equal(answers.delivered_on.value, '2026-09-28');
  assert.equal(answers.amount_paid.value, 701.1);
  assert.equal(answers.days_since_delivery.value, 11);
  assert.deepEqual(answers.damage.value, ['cracked screen', 'dented corner']);
  assert.equal(answers.tracking_number.value, null);
  assert.ok(answers.repair_cost.p10 <= answers.repair_cost.median && answers.repair_cost.median <= answers.repair_cost.p90);
});

test('estimates comparable property sales and forecasts TSA screenings', () => {
  const comps = `Comparable sales:
sqft,beds,year_built,neighborhood,sale_price
1450,3,1998,Elm Park,412000
2100,4,2006,Elm Park,585000
980,2,1975,Riverside,265000
1600,3,2003,Elm Park,449000
1300,3,1990,Riverside,318000

Subject property:
sqft,beds,year_built,neighborhood
1620,3,2001,Elm Park`;

  const { answers } = extract(comps, {
    sale_price: { type: 'estimate', unit: 'USD', min: 50000, max: 5000000, scale: 'log', step: 1000 },
  });
  assert.ok(answers.sale_price.p10 >= 350000 && answers.sale_price.p90 <= 500000);
  assert.equal(answers.sale_price.median % 1000, 0);
});
