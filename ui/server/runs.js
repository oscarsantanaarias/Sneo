import { AsyncLocalStorage } from 'async_hooks';
import { execFile } from 'child_process';
import { pushPending, updatePending } from './storage.js';

export const runStore = new AsyncLocalStorage();

const waiting = new Map();
const queues = new Map();

export const stoppedError = (message = 'Stopped by the user.') => Object.assign(new Error(message), {name: 'AbortError'});

export const trackChild = (child) => {
  const run = runStore.getStore();

  if(!run?.children || !child?.pid){
    return child;
  }

  if(run.signal?.aborted){
    endTree(child.pid);
    return child;
  }

  run.children.add(child);
  child.once('exit', () => run.children.delete(child));
  return child;
};

export const endTree = (pid) => new Promise(resolve => {
  if(!pid){
    return resolve(false);
  }

  if(process.platform === 'win32'){
    execFile('taskkill', ['/PID', String(pid), '/T', '/F'], {windowsHide: true, timeout: 15000}, err => resolve(!err));
    return;
  }

  try {
    process.kill(-pid, 'SIGKILL');
    return resolve(true);
  } catch(e) {
    try {
      process.kill(pid, 'SIGKILL');
    } catch(err) {
      return resolve(false);
    }

    return resolve(true);
  }
});

export const stopChildren = async (run) => {
  const children = [...(run?.children || [])];
  run?.children?.clear();
  return await Promise.all(children.map(child => endTree(child.pid)));
};

export const askApproval = async (item) => {
  const run = runStore.getStore();

  if(!run){
    return 'approved';
  }

  if(run.signal?.aborted){
    throw stoppedError();
  }

  const pending = await pushPending({...item, runId: run.id, chatId: run.chatId || ''});
  run.onEvent?.({type: 'approval', item: {id: pending.id, type: pending.type, title: pending.title}});

  return await new Promise((resolve, reject) => {
    const stop = () => {
      waiting.delete(pending.id);
      updatePending(pending.id, {status: 'cancelled'}).catch(() => {});
      reject(stoppedError('Stopped by the user while waiting for approval.'));
    };

    if(run.signal?.aborted){
      return stop();
    }

    run.signal?.addEventListener('abort', stop, {once: true});
    const done = (answer) => {
      run.signal?.removeEventListener('abort', stop);
      waiting.delete(pending.id);
      run.onEvent?.({type: 'approval-done', id: pending.id, answer});
      resolve(answer);
    };

    done.runId = run.id;
    waiting.set(pending.id, done);
  });
};

export const approveRun = async (runId) => {
  const mine = [...waiting].filter(([, done]) => done.runId === runId);

  for(const [id, done] of mine){
    await updatePending(id, {status: 'approved'}).catch(() => {});
    done('approved');
  }

  return mine.length;
};

export const answerApproval = (id, answer) => {
  const done = waiting.get(id);

  if(!done){
    return false;
  }

  done(answer === 'approved' ? 'approved' : 'denied');
  return true;
};

export class RunQueue {
  constructor(limit = 8){
    this.limit = Math.max(1, Number(limit) || 8);
    this.busy = false;
    this.waiting = [];
  }

  get full(){
    return this.busy && this.waiting.length >= this.limit;
  }

  enter(signal, onPosition){
    if(signal?.aborted){
      return Promise.reject(stoppedError());
    }

    if(!this.busy){
      this.busy = true;
      return Promise.resolve(this.exit());
    }

    if(this.waiting.length >= this.limit){
      return Promise.reject(Object.assign(new Error(`SNEO is busy: ${this.limit} requests are already waiting for the model. Try again in a minute.`), {status: 429}));
    }

    return new Promise((resolve, reject) => {
      const entry = {resolve, signal, onPosition, cancel: null};

      entry.cancel = () => {
        const index = this.waiting.indexOf(entry);

        if(index < 0){
          return;
        }

        this.waiting.splice(index, 1);
        this.announce();
        reject(stoppedError('Stopped before it started.'));
      };

      signal?.addEventListener('abort', entry.cancel, {once: true});
      this.waiting.push(entry);
      onPosition?.(this.waiting.length);
    });
  }

  exit(){
    let done = false;

    return () => {
      if(done){
        return;
      }

      done = true;
      const next = this.waiting.shift();

      if(!next){
        this.busy = false;
        return;
      }

      next.signal?.removeEventListener('abort', next.cancel);
      this.announce();
      next.resolve(this.exit());
    };
  }

  announce(){
    this.waiting.forEach((entry, index) => entry.onPosition?.(index + 1));
  }
}

export const queueFor = (key, limit) => {
  const name = String(key || 'default');

  if(!queues.has(name)){
    queues.set(name, new RunQueue(limit));
  }

  return queues.get(name);
};
