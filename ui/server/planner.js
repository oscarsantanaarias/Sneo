import { cleanText, makeId, readPlans, savePlans } from './storage.js';

export const planCreate = async ({title, steps}) => {
  const data = await readPlans();
  const plan = {
    id: makeId(),
    title: cleanText(title) || 'Plan',
    steps: (Array.isArray(steps) ? steps : []).map((step, index) => ({
      id: makeId(),
      text: cleanText(step.text || step),
      status: step.status || (index === 0 ? 'in_progress' : 'pending')
    })).filter(step => step.text),
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };

  data.plans.unshift(plan);
  data.plans = data.plans.slice(0, 100);
  await savePlans(data);

  return plan;
};

export const planList = async () => {
  const data = await readPlans();
  return data.plans.slice(0, 50);
};

export const planUpdate = async ({planId, stepId, status, text}) => {
  const data = await readPlans();
  const plan = data.plans.find(item => item.id === planId);

  if(!plan){
    throw new Error('plan not found');
  }

  const step = plan.steps.find(item => item.id === stepId);

  if(!step){
    throw new Error('step not found');
  }

  if(status) step.status = status;
  if(text !== undefined) step.text = cleanText(text);

  plan.updatedAt = new Date().toISOString();
  await savePlans(data);

  return plan;
};
