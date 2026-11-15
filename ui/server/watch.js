const searchTools = new Set(['search_files', 'grep', 'glob', 'pc_search', 'codebase_search', 're_search', 'web_search', 'context_pick', 'memory_search', 'vector_recall']);

export const guardLimits = {
  missStreak: 4,
  searches: 30,
  planSearches: 15,
  rewrites: 4,
  sameFailure: 3,
  context: 0.93
};

export const watchNotes = {
  'searches-failed': 'My last searches found nothing, so I stopped searching instead of guessing. Tell me where to look (a file, a folder or the exact name), or check the spelling.',
  searches: 'I ran a lot of searches in this turn and stopped so the chat does not fill up. This is what I have so far; tell me which lead to follow next.',
  rewrites: 'I kept rewriting the plan, so I stopped with the latest version. Tell me what to change, or run the plan as it is.',
  repeat: 'The same tool call kept failing with the same error, so I stopped instead of trying it again. Check the error above, or give me a different path, command or project.',
  context: 'This chat is almost out of room (over 93% of the model\'s context window), so I stopped. Start a new chat, or say "continue" and I will go on from a summary of the older messages.',
  empty: 'I stopped without writing an answer. Send your message again, or say "continue".',
  acted: 'I finished the steps shown above but did not write a final answer. Check the results above, or say "continue" and I will sum them up.'
};

const foundNothing = (result) => {
  if(!result || result.error){
    return true;
  }

  if(Array.isArray(result)){
    return result.length === 0;
  }

  if(typeof result === 'string'){
    return !result.trim() || /\b(no (results|matches)|0 matches|nothing found)\b/i.test(result);
  }

  const list = result.files || result.results || result.items || result.matches;
  return Array.isArray(list) ? list.length === 0 : false;
};

export class LoopWatch {
  constructor({mode = 'auto', window = 0, limits = {}} = {}){
    this.mode = mode;
    this.window = Number(window) || 0;
    this.limits = {...guardLimits, ...limits};
    this.searches = 0;
    this.emptyRow = 0;
    this.plans = 0;
    this.failures = new Map();
    this.reason = null;
  }

  stop(reason){
    this.reason = this.reason || reason;
    return {stop: this.reason, note: watchNotes[this.reason]};
  }

  tool(name, args, result){
    if(this.reason){
      return this.stop(this.reason);
    }

    if(name === 'plan_create'){
      this.plans++;

      if(this.plans >= this.limits.rewrites){
        return this.stop('rewrites');
      }
    }

    if(result?.error){
      const key = JSON.stringify([name, args || {}, String(result.error).slice(0, 300)]);
      const count = (this.failures.get(key) || 0) + 1;
      this.failures.set(key, count);

      if(count >= this.limits.sameFailure){
        return this.stop('repeat');
      }

      if(count === this.limits.sameFailure - 1){
        return {redirect: 'This exact call has now failed twice with the same error. Do not send it again: change the arguments, use another tool, or tell the user plainly what is blocking you.'};
      }
    }

    if(!searchTools.has(name)){
      return null;
    }

    this.searches++;
    this.emptyRow = foundNothing(result) ? this.emptyRow + 1 : 0;

    if(this.emptyRow >= this.limits.missStreak){
      return this.stop('searches-failed');
    }

    if(this.searches >= (this.mode === 'plan' ? this.limits.planSearches : this.limits.searches)){
      return this.stop('searches');
    }

    if(this.emptyRow === this.limits.missStreak - 1){
      return {redirect: 'Your last searches found nothing. Change the approach before searching again: list the files, read the likely file directly, or ask the user where it is.'};
    }

    return null;
  }

  text(answer){
    if(this.reason){
      return this.stop(this.reason);
    }

    if(!/(?:^|\n)[ \t]{0,3}1[.)][ \t]+\S/.test(String(answer || ''))){
      return null;
    }

    if(this.mode !== 'plan'){
      return null;
    }

    this.plans++;
    return this.plans >= this.limits.rewrites ? this.stop('rewrites') : null;
  }

  context(tokens){
    if(this.reason){
      return this.stop(this.reason);
    }

    if(!this.window || !(tokens > 0)){
      return null;
    }

    return tokens / this.window >= this.limits.context ? this.stop('context') : null;
  }
}

export const silentNote = (acted) => {
  return acted
    ? watchNotes.acted
    : watchNotes.empty;
};
