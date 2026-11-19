// el texto a la vista no tenia corte de bucles: un modelo trabado repetia el mismo tramo hasta el limite de salida
// aqui se mira si el final del texto es un tramo repetido y donde termina su primera copia

const letters = (text) => (text.match(/[A-Za-zÀ-￿]/g) || []).length;

const sameShare = (text, period, span) => {
  let same = 0;

  for(let i = text.length - span; i < text.length; i++){
    same += text[i] === text[i - period] ? 1 : 0;
  }

  return same / span;
};

// devuelve donde cortar para dejar una sola copia del tramo repetido, o -1 si el texto no esta trabado
export const loopStart = (text, {least = 1200, turns = 5, widest = 4000, sure = 0.98} = {}) => {
  const source = String(text ?? '');
  const probe = 80;

  if(source.length < least + probe){
    return -1;
  }

  const tail = source.slice(-probe);
  let from = source.length - probe - 1;

  for(let tries = 0; tries < 8 && from >= 0; tries++){
    const at = source.lastIndexOf(tail, from);

    if(at < 0){
      return -1;
    }

    const period = source.length - probe - at;
    const span = Math.max(least, period * turns);

    if(period > widest || span + period > source.length){
      return -1;
    }

    from = at - 1;

    if(sameShare(source, period, span) < sure || letters(source.slice(-span)) * 4 < span){
      continue;
    }

    let start = source.length - span;

    while(start > period && source[start - 1] === source[start - 1 - period]){
      start--;
    }

    return start;
  }

  return -1;
};
