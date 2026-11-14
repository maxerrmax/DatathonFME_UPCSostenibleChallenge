/* UPC Energy Intelligence v3 — plain JavaScript, no frontend build pipeline. */
'use strict';
const $ = (id) => document.getElementById(id);
const state = {dataset:'campus', datasets:[], view:'overview', overview:null, charts:[], aiCharts:[], busy:false, labDates:[],
  lastAi:null, lastLab:null, board:[], lastReply:null, radar:null, history:[], arena:null, studies:null, arenaTimer:null, lastExplore:null};
const ca = new Intl.NumberFormat('ca-ES',{maximumFractionDigits:1});
const eu = (v,dec=1) => v === null || v === undefined || !Number.isFinite(Number(v)) ? '—' : new Intl.NumberFormat('ca-ES',{minimumFractionDigits:dec,maximumFractionDigits:dec}).format(v);
const hh = (h) => String(h).padStart(2,'0');
const MONTHS = ['gen.','feb.','març','abr.','maig','juny','jul.','ag.','set.','oct.','nov.','des.'];
const COLORS = {orange:'#E77741', blue:'#4A6C9A', green:'#789A77', ink:'#23313D', grey:'#9aa7a6', light:'#c7d3cf', purple:'#9C76A9'};
const config = {responsive:true, displaylogo:false, displayModeBar:false, scrollZoom:false};
const graphTheme = {
  paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:'DM Sans, Arial, sans-serif',color:'#7d8d8e',size:11},
  margin:{l:52,r:24,t:16,b:38},
  xaxis:{showgrid:false,zeroline:false,tickfont:{size:10},linecolor:'#e7eae6'},
  yaxis:{gridcolor:'#edf0ed',zeroline:false,tickfont:{size:10}},
  showlegend:true,legend:{orientation:'h',y:1.16,x:0,font:{size:10}},hovermode:'x unified',
};
let toastTimer;
function toast(text){ const e=$('toast');e.textContent=text;e.classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>e.classList.remove('visible'),4200);}
async function api(path,options={}){
  let response;
  try{ response=await fetch(path,options); }
  catch(error){throw new Error('No es pot connectar al servidor. Comprova que uvicorn està en marxa.');}
  let result;
  try{result=await response.json();}catch{return Promise.reject(Error('Resposta inesperada del servidor'));}
  if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:'Error de consulta');
  return result;
}
const post = (path, body) => api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function el(tag, cls, text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;}
function emptyChart(node,msg){node.replaceChildren();node.append(el('div','empty-state',msg));}
// Plotly ≥ 3 only accepts {text: …} titles: string titles were silently dropped in v2.
const asTitle = (t) => typeof t === 'string' ? {text:t} : t;
function normalizeTitles(layout, data){
  Object.keys(layout).forEach(k=>{ if(/^[xy]axis\d*$/.test(k) && layout[k]?.title!==undefined) layout[k]={...layout[k],title:asTitle(layout[k].title)}; });
  if(layout.title!==undefined) layout.title=asTitle(layout.title);
  (data||[]).forEach(t=>{ if(t.colorbar?.title!==undefined) t.colorbar={...t.colorbar,title:asTitle(t.colorbar.title)}; });
}
function plot(id,data,layout={},height){
  const node=typeof id==='string'?$(id):id;
  if(!node||!window.Plotly)return;
  const l={...graphTheme,...layout}; if(height)l.height=height;
  normalizeTitles(l,data);
  Plotly.react(node,data,l,{...config});
}
function plotFromJson(node,figure){
  if (!window.Plotly) { emptyChart(node, 'La biblioteca de gràfics no s’ha carregat.'); return; }
  const base = Math.max(275,Math.min(390,window.innerHeight*.43));
  const h = figure.layout?.height ? Math.max(base, figure.layout.height) : base;
  Plotly.react(node,figure.data,{...figure.layout,height:h,autosize:true},{...config,responsive:true})
    .then(()=>Plotly.Plots.resize(node))
    .catch(err=>{console.error('Error renderitzant Plotly:',err);emptyChart(node,'No s’ha pogut mostrar el gràfic.');});
}
function figureOf(id){const g=$(id);return g?.data?.length?{data:JSON.parse(JSON.stringify(g.data)),layout:JSON.parse(JSON.stringify(g.layout))}:null;}
function band(x, lo, hi, color, yaxis){
  return [{x, y:hi, type:'scatter', mode:'lines', line:{width:0}, showlegend:false, hoverinfo:'skip', yaxis},
          {x, y:lo, type:'scatter', mode:'lines', line:{width:0}, fill:'tonexty', fillcolor:color, showlegend:false, hoverinfo:'skip', yaxis}];
}
function frozenShapes(hours, yref='paper'){
  return (hours||[]).map(h=>({type:'rect',xref:'x',yref,x0:h-.5,x1:h+.5,y0:0,y1:1,fillcolor:'rgba(130,140,150,.12)',line:{width:0},layer:'below'}));
}

/* ---------- charts list (Explorer, Copilot) ---------- */
function plotCards(containerId,charts,origin,insights,specFor){
  const container=$(containerId);container.replaceChildren();
  if(!charts?.length){emptyChart(container,'No s’han generat gràfics.');return;}
  charts.forEach((c,i)=>{
    const wrapper=el('div','chart-entry');
    const heading=el('div','chart-entry-head');
    const title=el('h4','chart-title',hh(i+1)+' / '+c.title);
    if(c.evidence){const chip=el('span','ev-chip',c.evidence);title.prepend(chip);}
    heading.append(title);
    const actions=el('div','chart-entry-actions');
    const pin=el('button','chart-mini-action','＋ Fixar al board');pin.type='button';pin.title='Afegir aquest gràfic al Decision Board';
    pin.addEventListener('click',()=>pinChart(c,origin,insights(),specFor?specFor(c):null));
    const download=el('button','chart-mini-action subtle','↧ PNG');download.type='button';
    download.addEventListener('click',()=>{const gd=wrapper.querySelector('.js-plotly-plot');if(gd) Plotly.downloadImage(gd,{format:'png',width:1400,height:820,filename:'upc-figure-'+(i+1)});});
    actions.append(pin,download);heading.append(actions);wrapper.append(heading);
    const holder=el('div','chart-plot-holder');wrapper.append(holder);container.append(wrapper);
    plotFromJson(holder,c.figure);
  });
}
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o);}
function dataset(){return state.datasets.find(d=>d.id===state.dataset) || state.datasets[0];}
function setView(name){
  state.view=name;
  document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id==='view-'+name));
  document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x.dataset.view===name));
  $('breadcrumb-current').textContent={overview:'Vista general',explore:'Data explorer',copilot:'AI Copilot',lab:'Decision lab',arena:'Agent arena',board:'Decision board'}[name];
  document.querySelector('.sidebar').classList.remove('open');$('mobile-backdrop').classList.remove('visible');
  setTimeout(()=>{document.querySelectorAll('.view.active .js-plotly-plot').forEach(x=>Plotly.Plots.resize(x));},80);
  if(name==='lab'){if(!state.labDates.length)loadDays();else runLab();loadRadar();}
  if(name==='explore' && !state.charts.length)runExplore();
  if(name==='arena')loadArena();
  if(name==='board')renderBoard();
  window.scrollTo({top:0,behavior:'smooth'});
}
function fillMetrics(){
  const d=dataset(); if(!d)return;
  for(const select of [$('explore-metric'),$('explore-second')]){
    const old=select.value;select.replaceChildren();
    if(select.id==='explore-second')option(select,'','Cap');
    d.metrics.forEach(m=>option(select,m.key,`${m.label}${m.unit?' · '+m.unit:''}`));
    if([...select.options].some(o=>o.value===old))select.value=old;
  }
  if(d.metrics.some(m=>m.key==='precio_eur_per_kwh'))$('explore-metric').value='precio_eur_per_kwh';
  if(d.metrics.some(m=>m.key==='co2_kg_per_kwh'))$('explore-second').value='co2_kg_per_kwh';
  $('data-name').textContent=d.name;$('data-source').textContent=d.source;
  $('data-rows').textContent=ca.format(d.rows);$('data-cols').textContent=ca.format(d.cols);
  $('data-missing').textContent=eu(d.missing_pct,1)+'%';
  if(!d.start)$('explore-kind').value='histogram';
  // Only months that exist in this dataset (train has no November).
  for(const sel of [$('filter-month'),$('radar-month')]){
    const keep=sel.value;sel.replaceChildren();option(sel,'','Tots els mesos');
    (d.months||[]).forEach(m=>option(sel,String(m),MONTHS[m-1]));
    if([...sel.options].some(o=>o.value===keep))sel.value=keep;
  }
  $('nav-item-arena')?.toggleAttribute('disabled',!d.official);
}
function buildDatasetSelect(){
  const selector=$('dataset-select');selector.replaceChildren();
  state.datasets.forEach(d=>option(selector,d.id,d.name));
  if(!state.datasets.some(d=>d.id===state.dataset))state.dataset=state.datasets[0]?.id||'campus';
  selector.value=state.dataset;fillMetrics();
}
async function init(){
  try{
    const status=await api('/api/status');state.datasets=status.datasets;
    $('mode-label').textContent=status.mode==='helmcode'?'HELMCODE CONFIGURAT':'MODE LOCAL';
    $('mode-sub').textContent=status.mode==='helmcode'?'Connexió pendent de verificar':'Analista local amb eines verificades';
    $('ai-mode-state').textContent=status.mode==='helmcode'?'Helmcode configurat · pendent de prova':'Analista local · eines i verificació actives';
    if(!state.datasets.length){toast('Falten les dades inicials. Importa un fitxer CSV per començar.');return;}
    buildDatasetSelect();await Promise.all([loadOverview(),loadDays(),loadQuality()]);
  }catch(e){toast(e.message);}
}

/* ---------- Overview ---------- */
async function loadQuality(){
  const box=$('quality-items');
  try{
    const q=await api('/api/quality?dataset_id='+encodeURIComponent(state.dataset));
    box.replaceChildren();
    if(!q.issues?.length){box.append(el('p','quality-ok','No s’han detectat incidències de qualitat rellevants.'));return;}
    q.issues.forEach(i=>{
      const card=el('div','quality-item level-'+i.level.normalize('NFD').replace(/[̀-ͯ]/g,''));
      card.append(el('span','quality-level',i.level.toUpperCase()),el('strong',null,i.title),el('p',null,i.detail),el('small',null,'→ '+i.action));
      box.append(card);
    });
  }catch(e){box.replaceChildren(el('p','quality-ok','Qualitat no disponible: '+e.message));}
}
async function loadOverview(){
  try{
    const month=$('filter-month').value;const typ=$('filter-daytype').value;
    const q=new URLSearchParams({dataset_id:state.dataset,day_type:typ});if(month)q.set('month',month);
    const x=await api('/api/overview?'+q);state.overview=x;
    const k=x.kpi;$('kpi-load').textContent=eu(k.demand_mwh,1);$('kpi-solar').textContent=eu(k.solar_mwh,1);
    $('kpi-co2').textContent=eu(k.emissions_t,1);$('kpi-cost').textContent=eu(k.cost_eur,0);
    $('correlation-chip').textContent='r = '+eu(x.corr,3);
    $('period-label').textContent=x.rows.toLocaleString('ca-ES')+' hores / registres';
    const r=x.robust||{}; const hr=[...Array(24).keys()];
    if(r.load_kw){
      const data=[];
      [['load_kw','Consum',COLORS.orange,'rgba(231,119,65,.13)'],['pv_kw','Solar',COLORS.green,'rgba(120,154,119,.15)'],['net_kw','Demanda neta',COLORS.blue,'rgba(74,108,154,.10)']].forEach(([m,name,color,fill])=>{
        if(!r[m])return;
        data.push(...band(hr,r[m].q25,r[m].q75,fill));
        data.push({x:hr,y:r[m].median,name:name+' · mediana',type:'scatter',mode:'lines',line:{color,width:2.6},hovertemplate:'%{y:.0f} kW'});
      });
      const shapes=frozenShapes(x.frozen_hours);
      const ann=x.frozen_hours?.length?[{x:2.5,y:0.04,yref:'paper',xanchor:'center',yanchor:'bottom',showarrow:false,text:'lectura<br>congelada',font:{size:10,color:'#7d8d8e'}}]:[];
      plot('chart-overview-load',data,{showlegend:false,shapes,annotations:ann,xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora del dia'},yaxis:{...graphTheme.yaxis,title:'kW',rangemode:'tozero'}});
      if(r.precio_eur_per_kwh&&r.co2_kg_per_kwh){
        const pr=r.precio_eur_per_kwh, co=r.co2_kg_per_kwh;
        const mdata=[...band(hr,pr.q25,pr.q75,'rgba(231,119,65,.14)','y'),
          {x:hr,y:pr.median,name:'Preu · mediana (€/kWh)',type:'scatter',mode:'lines+markers',line:{color:COLORS.orange,width:2.6},marker:{size:4},yaxis:'y',hovertemplate:'%{y:.4f} €/kWh'},
          ...band(hr,co.q25,co.q75,'rgba(74,108,154,.14)','y2'),
          {x:hr,y:co.median,name:'CO₂ · mediana (kg/kWh)',type:'scatter',mode:'lines+markers',line:{color:COLORS.blue,width:2.6},marker:{size:4},yaxis:'y2',hovertemplate:'%{y:.4f} kg/kWh'}];
        plot('chart-overview-market',mdata,{height:330,xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora del dia',anchor:'y2'},
          yaxis:{...graphTheme.yaxis,title:'€/kWh',domain:[.56,1]},yaxis2:{...graphTheme.yaxis,title:'kg CO₂/kWh',domain:[0,.44]}});
      }
    }else{emptyChart($('chart-overview-load'),'Aquest dataset no té hores identificades.');emptyChart($('chart-overview-market'),'Carrega dades amb data i hora.');}
    const box=$('overview-insights');box.replaceChildren();(x.recommended||[]).forEach((txt,i)=>{const item=el('div','insight-item');item.append(el('i',null,'0'+(i+1)),el('p',null,txt));box.append(item);});
  }catch(e){toast(e.message);}
}

/* ---------- Explorer ---------- */
async function runExplore(){
  const m1=$('explore-metric').value;const m2=$('explore-second').value;
  const req={dataset_id:state.dataset,kind:$('explore-kind').value,metrics:[m1,...(m2&&m2!==m1?[m2]:[])],group:$('explore-group').value,aggregation:$('explore-agg').value};
  $('explore-run').disabled=true;$('explore-run').textContent='Generant…';
  try{const x=await post('/api/chart',req);state.charts=x.charts;state.lastExplore=req;
    plotCards('explore-charts',x.charts,'Data Explorer',()=>['Gràfic configurat al Data Explorer: '+req.kind+' · '+req.metrics.join(', ')+' · '+req.aggregation+' per '+req.group+'.'],()=>({type:'chart',req}));
    $('explore-title').textContent=x.charts.map(c=>c.title).join(' / ');}
  catch(e){toast(e.message);emptyChart($('explore-charts'),e.message);}
  finally{$('explore-run').disabled=false;$('explore-run').innerHTML='Generar visualització <svg><use href="#i-arrow"/></svg>';}
}

/* ---------- Copilot ---------- */
function addChat(role,lines){
  const node=el('div','message '+role);
  node.append(el('div','message-label',role==='user'?'TU':'ENERGY COPILOT'));
  (Array.isArray(lines)?lines:[lines]).forEach(x=>{
    // Evidence chips [E1] and **bold** are built as DOM nodes (never innerHTML): model text stays inert.
    const line=String(x).replace(/^#+\s*/,'').replace(/^[-*]\s+/,'• ');
    const p=el('p'); const parts=line.split(/(\[E\d+\]|\*\*[^*]+\*\*)/g);
    parts.forEach(part=>{
      if(/^\[E\d+\]$/.test(part)) p.append(el('span','ev-chip',part.slice(1,-1)));
      else if(/^\*\*[^*]+\*\*$/.test(part)) p.append(el('strong',null,part.slice(2,-2)));
      else if(part) p.append(document.createTextNode(part));
    });
    if(/^\**Decisió:/.test(line))p.classList.add('decision-line');
    node.append(p);
  });
  $('chat-messages').append(node);$('chat-messages').scrollTop=$('chat-messages').scrollHeight;return node;
}
const MODE_LABEL={greeting:'Presentació',helmcode:'Planificat amb Helmcode',context:'Context anterior verificat',physics:'Simulació física verificada',local:'Planificador local',
  'analyst-helmcode':'Analista Helmcode · eines verificades','analyst-helmcode-json':'Analista Helmcode (pla JSON) · eines verificades','analyst-local':'Analista local · eines verificades'};
async function askAssistant(question){
  if(state.busy)return;
  state.busy=true;$('chat-send').disabled=true;
  addChat('user',question);$('chat-input').value='';setView('copilot');
  const loading=addChat('assistant','Triant eines, calculant i verificant les xifres…');
  try{
    const context=state.lastAi?.dataset===state.dataset ? state.lastAi : null;
    const payload={message:question,dataset_id:state.dataset,
      previous_query:context?.question||null,previous_plan:context?.plan||null,history:state.history.slice(-3)};
    const res=await post('/api/assistant',payload);
    loading.remove();
    let messages=res.answer?res.answer.split('\n').filter(Boolean):(res.insights.length?[...res.insights]:['No he trobat dades suficients per extreure una conclusió.']);
    if(res.llm_warning){
      messages=['Avís: '+res.llm_warning,...messages];
      $('ai-mode-state').textContent='Mode local de reserva · consulta els detalls';$('mode-label').textContent='FALLBACK LOCAL';
    }else if(res.mode==='helmcode'||String(res.mode).startsWith('analyst-helmcode')){
      $('ai-mode-state').textContent='Helmcode verificat · càlcul local';$('mode-label').textContent='HELMCODE VERIFICAT';
    }else if(res.mode==='context'){$('ai-mode-state').textContent='Context reutilitzat · càlcul verificat';}
    addChat('assistant',messages);
    state.lastReply=res;
    state.history.push({q:question,a:messages.join('\n').slice(0,900)});state.history=state.history.slice(-3);
    if(res.plan?.kind && !['analysis','what_if'].includes(res.plan.kind))state.lastAi={dataset:state.dataset,question,plan:res.plan};
    $('copilot-context').hidden=!state.lastAi;
    if(state.lastAi){$('context-label').textContent='Context de gràfic · '+state.lastAi.plan.metrics.map(m=>dataset()?.metrics.find(x=>x.key===m)?.label||m).join(' + ');}
    if(res.mode==='greeting'){$('ai-evidence').hidden=true;return;}
    state.aiCharts=res.charts;
    plotCards('ai-canvas-charts',res.charts,'AI Copilot',()=>messages.filter(m=>!m.startsWith('Avís')).map(m=>m.replace(/\s*\[E\d+\]/g,'').replace(/\*\*/g,'')).slice(0,4),
      ()=>({type:'question',question}));
    $('ai-canvas-title').textContent=res.charts.map(c=>c.title).join(' / ')||'Resposta sense gràfic';
    $('ai-canvas-subtitle').textContent=(MODE_LABEL[res.mode]||res.mode)+' · '+ca.format(res.evidence?.rows||0)+' registres';
    renderEvidence(res);
  }catch(e){loading.remove();addChat('assistant','No s’ha pogut completar l’anàlisi: '+e.message);toast(e.message);}
  finally{state.busy=false;$('chat-send').disabled=false;}
}
function renderEvidence(res){
  $('ai-evidence').hidden=false;
  const trace=$('ai-trace');trace.replaceChildren();
  (res.trace||[]).forEach(t=>{
    const li=el('li','trace-item'+(t.error?' error':''));
    const head=el('div','trace-head');head.append(el('span','ev-chip',t.id||'—'),el('code',null,t.tool+'('+Object.entries(t.args||{}).filter(([,v])=>v!==null&&v!==undefined).map(([k,v])=>k+'='+JSON.stringify(v)).join(', ')+')'));
    li.append(head);
    if(t.error)li.append(el('p',null,'Error: '+t.error));
    trace.append(li);
  });
  const v=res.verification, badge=$('ai-verify');
  if(v){badge.textContent=v.ok?`✓ ${v.verified}/${v.checked} xifres verificades`:`⚠ ${v.unverified.length} xifres sense evidència: ${v.unverified.join(', ')}`;badge.className='verify-badge '+(v.ok?'ok':'warn');}
  else{badge.textContent='càlcul directe al servidor';badge.className='verify-badge ok';}
  const text=$('ai-evidence-text');text.replaceChildren(el('p',null,res.note||''));
  if(res.plan?.kind && res.plan.kind!=='analysis')text.append(el('p',null,'Operació: '+res.plan.kind+' · agregació: '+res.plan.aggregation+' · variables: '+(res.plan.metrics||[]).join(', ')));
}

/* ---------- Radar ---------- */
async function loadRadar(){
  try{
    const m=$('radar-month').value;
    const result=await api('/api/radar?dataset_id='+encodeURIComponent(state.dataset)+(m?'&month='+m:''));
    state.radar=result; const c=result.candidate;
    $('radar-charge').textContent=hh(c.charge_hour)+':00';$('radar-discharge').textContent=hh(c.discharge_hour)+':00';
    $('radar-reliability').textContent=c.viable_day_pct==null?'—':eu(c.viable_day_pct,1)+'%';
    $('radar-note').textContent=result.disclaimer+' Mostra: '+c.days_examined+' dies complets (els mateixos que el Lab).';
    const ticks=result.profile.map(x=>x.hour),values=result.profile.map(x=>x.signal);
    const colors=ticks.map(h=>h===c.charge_hour?'#79a18a':h===c.discharge_hour?'#db7d53':'#bfcfca');
    plot('radar-chart',[{type:'bar',x:ticks,y:values,marker:{color:colors},name:'Senyal combinat',hovertemplate:'%{x}:00 · índex %{y:.3f}<extra></extra>'}],
      {height:265,showlegend:false,margin:{l:48,r:12,t:18,b:38},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{dash:'dot',color:'#9aa7a6',width:1}}],
       xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora del dia'},yaxis:{...graphTheme.yaxis,title:'Índex (mitjana = 1)'}},265);
    const q=c.spread_quartiles||[];
    plot('radar-spread',[{type:'histogram',x:result.day_spreads,nbinsx:30,marker:{color:'#c9d6d1',line:{color:'#fff',width:1}},hovertemplate:'marge %{x}<br>%{y} dies<extra></extra>'}],
      {height:265,showlegend:false,margin:{l:44,r:12,t:18,b:38},bargap:.04,
       shapes:[{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,line:{color:'#db7d53',width:2}},...(q.length?[{type:'line',x0:q[2],x1:q[2],yref:'paper',y0:0,y1:1,line:{color:'#4A6C9A',dash:'dot',width:2}}]:[])],
       annotations:[{x:0,y:1,yref:'paper',text:'marge 0',showarrow:false,xanchor:'right',font:{size:10,color:'#db7d53'}},...(q.length?[{x:q[2],y:.92,yref:'paper',text:'mediana',showarrow:false,xanchor:'left',font:{size:10,color:'#4A6C9A'}}]:[])],
       xaxis:{...graphTheme.xaxis,title:'Marge diari del parell '+hh(c.charge_hour)+'→'+hh(c.discharge_hour)+' h'},yaxis:{...graphTheme.yaxis,title:'Dies'}},265);
    const mx=result.matrix;
    if(mx?.months?.length){
      const ylab=mx.months.map(m=>MONTHS[m-1]);
      const best=result.by_month||[];
      plot('radar-matrix',[
        {type:'heatmap',z:mx.z,x:[...Array(24).keys()],y:ylab,colorscale:[[0,'#eef3ef'],[.45,'#d8c5a8'],[.75,'#d77a50'],[1,'#743a26']],colorbar:{title:'índex',thickness:10},xgap:1,ygap:1,hovertemplate:'%{y} · %{x}:00<br>índex %{z:.2f}<extra></extra>'},
        {type:'scatter',mode:'markers',x:best.map(b=>b.charge_hour),y:best.map(b=>MONTHS[b.month-1]),marker:{size:11,color:'rgba(0,0,0,0)',line:{color:'#2f6b48',width:2.5}},name:'Càrrega',hovertemplate:'càrrega %{x}:00<extra>%{y}</extra>'},
        {type:'scatter',mode:'markers',x:best.map(b=>b.discharge_hour),y:best.map(b=>MONTHS[b.month-1]),marker:{size:11,color:'rgba(0,0,0,0)',line:{color:'#23313D',width:2.5}},name:'Descàrrega',hovertemplate:'descàrrega %{x}:00<extra>%{y}</extra>'}],
        {height:330,margin:{l:52,r:12,t:26,b:38},hovermode:'closest',xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora del dia'},yaxis:{...graphTheme.yaxis,autorange:'reversed'}},330);
    }
  }catch(e){$('radar-note').textContent='Radar no disponible: '+e.message;}
}

/* ---------- Decision Lab ---------- */
async function loadDays(){
  const select=$('lab-date');
  try{
    const res=await api('/api/days?dataset_id='+encodeURIComponent(state.dataset));state.labDates=res.dates;
    select.replaceChildren();if(!res.dates.length){option(select,'','Sense dies complets');toast(res.reason||'Aquest dataset no és apte per simular la bateria.');return;}
    res.dates.forEach(d=>option(select,d,d+' · '+['dl','dt','dc','dj','dv','ds','dg'][(new Date(d+'T12:00:00').getDay()+6)%7]));
    select.value=res.default||res.dates[res.dates.length-1];
    if(state.view==='lab')await runLab();
  }catch(e){toast(e.message);}
}
function windowLabel(a,b){return hh(a)+'–'+hh(b)+' h';}
function syncWindows(){
  const cs=+$('lab-charge-start').value,ce=+$('lab-charge-end').value,ds=+$('lab-discharge-start').value,de=+$('lab-discharge-end').value;
  $('lab-charge-value').textContent=windowLabel(cs,ce);$('lab-discharge-value').textContent=windowLabel(ds,de);
}
async function runLab(){
  if(!$('lab-date').value)return;
  const req={dataset_id:state.dataset,date:$('lab-date').value,charge_start:+$('lab-charge-start').value,charge_end:+$('lab-charge-end').value,
    discharge_start:+$('lab-discharge-start').value,discharge_end:+$('lab-discharge-end').value,power_kw:+$('lab-power').value};
  $('lab-run').disabled=true;
  try{
    const x=await post('/api/simulate',req);
    $('lab-base-cost').textContent=eu(x.baseline.cost,0)+' €';
    $('lab-man-score').textContent=eu(x.manual.score,2)+' p';
    $('lab-opt-score').textContent=eu(x.optimal.score,2)+' p';
    $('lab-capture').textContent=x.capture_pct==null?'—':eu(x.capture_pct,0)+'%';
    $('lab-capture-bar').style.width=Math.max(0,Math.min(100,x.capture_pct||0))+'%';
    $('lab-notice').textContent=x.notice;
    $('lab-window-hint').textContent=x.overlap_hours.length?'⚠ Les finestres es solapen a '+x.overlap_hours.map(hh).join(', ')+' h: allà es descarrega.':'Finestres inclusives; poden creuar la mitjanit (p. ex. 22 → 03).';
    state.lastLab={response:x, req};
    const bar=x.manual_actions.map(a=>a>=0?'#7E9F83':'#E77B50');
    const shade=[...x.charge_hours.map(h=>({type:'rect',xref:'x',yref:'paper',x0:h-.5,x1:h+.5,y0:0,y1:1,fillcolor:'rgba(126,159,131,.10)',line:{width:0},layer:'below'})),
                 ...x.discharge_hours.map(h=>({type:'rect',xref:'x',yref:'paper',x0:h-.5,x1:h+.5,y0:0,y1:1,fillcolor:'rgba(231,123,80,.09)',line:{width:0},layer:'below'}))];
    plot('lab-dispatch-chart',[
      {x:x.hours,y:x.manual_actions,type:'bar',name:'La teva consigna · kW',marker:{color:bar},opacity:.9},
      {x:x.hours,y:x.optimal_actions,type:'scatter',mode:'markers',name:'Consigna òptima · kW',marker:{symbol:'line-ew-open',size:16,color:COLORS.ink,line:{width:2}}},
      {x:x.hours,y:x.manual_soc,type:'scatter',mode:'lines+markers',name:'SOC · kWh',yaxis:'y2',line:{color:'#517698',width:2.6},marker:{size:4}},
      {x:x.hours,y:x.optimal_soc,type:'scatter',mode:'lines',name:'SOC òptim · kWh',yaxis:'y2',line:{color:'#A9B4AC',width:2,dash:'dot'}},
    ],{height:350,shapes:shade,barmode:'relative',xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora del dia'},yaxis:{...graphTheme.yaxis,title:'Potència (kW)',zeroline:true,zerolinecolor:'#CCD6D1',range:[-560,560]},yaxis2:{title:'SOC (kWh)',overlaying:'y',side:'right',range:[0,2100],showgrid:false},legend:{orientation:'h',y:1.17,x:0}},350);
    const optColor=x.optimal_actions.map(a=>a>1?'#79a18a':a<-1?'#db7d53':'#c7d3cf');
    plot('lab-why-chart',[
      {x:x.hours,y:x.weights,type:'bar',name:'Cost de score d’importar 1 kWh',marker:{color:optColor},hovertemplate:'%{x}:00 · %{y:.4f} punts/kWh<extra></extra>'},
      {x:x.hours,y:x.optimal_actions,type:'scatter',mode:'lines+markers',name:'Consigna òptima (kW)',yaxis:'y2',line:{color:COLORS.ink,width:2,shape:'hv'},marker:{size:4}}],
      {height:330,xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora del dia',anchor:'y2'},yaxis:{...graphTheme.yaxis,title:'punts / kWh',domain:[.46,1]},
       yaxis2:{...graphTheme.yaxis,title:'kW',domain:[0,.38],zeroline:true,zerolinecolor:'#CCD6D1'},legend:{orientation:'h',y:1.12,x:0}},330);
    const table=$('lab-result-table');table.replaceChildren();
    const dom=el('table');const thead=el('thead');const tr=el('tr');
    ['Mètrica','La teva decisió','Òptim retrospectiu'].forEach(t=>tr.append(el('th',null,t)));thead.append(tr);dom.append(thead);
    const rows=[['Score del dia (fórmula oficial)',eu(x.manual.score,2)+' p',eu(x.optimal.score,2)+' p'],['CO₂ estalviat',eu(x.manual.co2_saved_kg,1)+' kg',eu(x.optimal.co2_saved_kg,1)+' kg'],['Cost estalviat',eu(x.manual.cost_saved_eur,2)+' €',eu(x.optimal.cost_saved_eur,2)+' €'],['Reducció d’emissions',eu(x.manual.co2_saved_pct,2)+'%',eu(x.optimal.co2_saved_pct,2)+'%'],['Reducció del cost',eu(x.manual.cost_saved_pct,2)+'%',eu(x.optimal.cost_saved_pct,2)+'%'],['Infraccions',String(x.manual.infractions),String(x.optimal.infractions)],['SOC final',eu(x.manual.soc_end_kwh,1)+' kWh',eu(x.optimal.soc_end_kwh,1)+' kWh']];
    const body=el('tbody');rows.forEach(r=>{const row=el('tr');r.forEach(v=>row.append(el('td',null,v)));body.append(row);});dom.append(body);table.append(dom);
  }catch(e){toast(e.message);}finally{$('lab-run').disabled=false;}
}

/* ---------- Agent Arena ---------- */
const POLICY_COLOR={idle:'#c7d3cf',rule:'#9aa7a6',team:COLORS.orange,mpc_naive:COLORS.blue,mpc_perfect:'#7d9cc4',oracle:COLORS.ink};
async function loadArena(){
  clearTimeout(state.arenaTimer);
  if(!dataset()?.official){$('arena-status-text').textContent='L’Arena només s’executa sobre les dades oficials del campus (selecciona el dataset de train).';return;}
  try{
    const [a,s]=await Promise.all([api('/api/arena'),api('/api/studies')]);
    if(a.status==='ready'&&a.result){state.arena=a.result;renderArena(a.result);}
    if(s.status==='ready'&&s.result){state.studies=s.result;renderStudies(s.result);}
    const running=[a,s].filter(x=>x.status==='running');
    if(a.status==='error'||s.status==='error')$('arena-status-text').textContent='Error: '+(a.error||s.error);
    else if(running.length){$('arena-status-text').textContent='Calculant amb el simulador oficial… '+(running[0].progress||'')+' ('+running[0].elapsed_s+' s)';state.arenaTimer=setTimeout(loadArena,1500);}
    else $('arena-status-text').textContent='Backtest complet · '+state.arena.weeks.length+' setmanes · agent: '+(state.arena.agent_path?state.arena.agent_path.split(/[\\/]/).slice(-2).join('/'):'cap')+(state.arena.agent_note?' · '+state.arena.agent_note:'');
  }catch(e){$('arena-status-text').textContent='Arena no disponible: '+e.message;}
}
function renderArena(r){
  const pols=r.policies.filter(p=>p.key!=='idle');
  const by=Object.fromEntries(r.policies.map(p=>[p.key,p]));
  const best=[...pols].sort((a,b)=>b.pooled.score-a.pooled.score)[0];
  const kpis=$('arena-kpis');kpis.replaceChildren();
  const card=(label,value,sub,accent)=>{const c=el('div','arena-kpi'+(accent?' accent':''));c.append(el('span',null,label),el('strong',null,value),el('small',null,sub));kpis.append(c);};
  card('MILLOR POLÍTICA',eu(best.pooled.score,2)+' p',best.label,true);
  if(by.team)card('BATTERYAGENT ACTUAL',eu(by.team.pooled.score,2)+' p',by.team.pooled.infractions+' infraccions · '+eu(by.team.pooled.score_statement_penalty,2)+' p amb penalització de l’enunciat');
  if(by.oracle)card('SOSTRE TEÒRIC',eu(by.oracle.pooled.score,2)+' p','Informació perfecta de tota la setmana');
  if(by.team&&by.oracle)card('MARGE DE MILLORA',eu(by.oracle.pooled.score-by.team.pooled.score,2)+' p','Del BatteryAgent actual a l’òptim');
  const order=[...pols].sort((a,b)=>a.pooled.score-b.pooled.score);
  plot('arena-score-chart',[
    {type:'bar',orientation:'h',y:order.map(p=>p.label),x:order.map(p=>p.pooled.score_statement_penalty),name:'Penalització ENUNCIAT (1 p/infracció)',marker:{color:'rgba(200,208,205,.55)'},hovertemplate:'%{x:.2f} p<extra>enunciat</extra>'},
    {type:'bar',orientation:'h',y:order.map(p=>p.label),x:order.map(p=>p.pooled.score),name:'Score environment.py',marker:{color:order.map(p=>POLICY_COLOR[p.key])},text:order.map(p=>eu(p.pooled.score,2)),textposition:'outside',cliponaxis:false,hovertemplate:'%{x:.2f} p<extra>oficial</extra>'}],
    {height:330,barmode:'overlay',margin:{l:300,r:56,t:20,b:38},hovermode:'closest',xaxis:{...graphTheme.xaxis,title:'Punts (0,7·ΔCO₂% + 0,3·ΔCost% − penalitzacions)'},yaxis:{...graphTheme.yaxis,automargin:true,tickfont:{size:10}}},330);
  const weeks=r.weeks.map(w=>MONTHS[w.month-1]+' '+w.id.slice(8));
  plot('arena-week-chart',pols.map(p=>({type:'scatter',mode:'lines+markers',x:weeks,y:p.weeks.map(w=>w.score),name:p.label.split(' (')[0],
    line:{color:POLICY_COLOR[p.key],width:p.key==='team'?2.6:1.3,dash:p.key==='oracle'?'dot':'solid'},marker:{size:p.key==='team'?8:6},opacity:p.key==='mpc_perfect'?.6:1})),
    {height:330,hovermode:'x unified',xaxis:{...graphTheme.xaxis,type:'category',title:'Setmana pseudo-test (mes i dia d’inici)'},yaxis:{...graphTheme.yaxis,title:'Score setmanal',rangemode:'tozero'},legend:{orientation:'h',y:1.22,x:0,font:{size:9}}},330);
  if(by.oracle){
    const steps=r.regret;const SHORT=['Miopia de 24 h','Previsió de consum','Política de l’agent'];const labels=['Òptim setmanal',...steps.map((s,i)=>SHORT[i]||('Pas '+(i+1))),by.team?'BatteryAgent actual':'Final'];
    plot('arena-regret-chart',[{type:'waterfall',orientation:'v',x:labels,measure:['absolute',...steps.map(()=>'relative'),'total'],
      y:[by.oracle.pooled.score,...steps.map(s=>-s.points),0],text:[eu(by.oracle.pooled.score,2),...steps.map(s=>'−'+eu(s.points,2)),eu(by.team?.pooled.score??0,2)],textposition:'outside',
      decreasing:{marker:{color:COLORS.orange}},increasing:{marker:{color:COLORS.green}},totals:{marker:{color:COLORS.ink}},connector:{line:{color:'#c7d3cf'}}}],
      {height:300,showlegend:false,hovermode:'closest',yaxis:{...graphTheme.yaxis,title:'Punts',rangemode:'tozero'},xaxis:{...graphTheme.xaxis,tickfont:{size:10}}},300);
    const call=$('arena-callout');call.replaceChildren();
    steps.forEach(s=>{const p=el('p');p.append(el('strong',null,s.step+': '+eu(s.points,2)+' punts. '),document.createTextNode(s.meaning));call.append(p);});
    const forecast=steps.find(s=>s.step.startsWith('Consum futur'));
    if(forecast&&forecast.points<0.1)call.append(el('p','arena-insight','→ Conclusió: la previsió de consum gairebé no aporta valor (la demanda neta mínima, '+eu(r.min_net_load_kw,0)+' kW, gairebé mai limita la descàrrega de 500 kW). L’esforç s’ha de posar en l’optimització amb les previsions de preu i CO₂, no en predir consum.'));
  }
  const table=$('arena-table');table.replaceChildren();
  const dom=el('table');const tr=el('tr');['Política','Score','Enunciat','ΔCO₂','ΔCost','CO₂','Infr.'].forEach(t=>tr.append(el('th',null,t)));
  const thead=el('thead');thead.append(tr);dom.append(thead);const body=el('tbody');
  [...pols].sort((a,b)=>b.pooled.score-a.pooled.score).forEach(p=>{const row=el('tr',p.key==='team'?'row-team':'');[p.label,eu(p.pooled.score,2),eu(p.pooled.score_statement_penalty,2),eu(p.pooled.co2_saved_pct,2)+'%',eu(p.pooled.cost_saved_pct,2)+'%',eu(p.pooled.co2_saved_t,1)+' t',String(p.pooled.infractions)].forEach(v=>row.append(el('td',null,v)));body.append(row);});
  dom.append(body);table.append(dom);
  $('arena-protocol').textContent=r.protocol+' '+r.leakage_warning+' '+r.penalty_note;
}
function renderStudies(s){
  const par=s.pareto;
  plot('studies-pareto',[{type:'scatter',mode:'lines+markers+text',x:par.map(p=>p.cost_saved_pct),y:par.map(p=>p.co2_saved_pct),text:par.map(p=>'α='+eu(p.alpha,1)),textposition:'top right',textfont:{size:9},
    marker:{size:par.map(p=>Math.abs(p.alpha-.7)<1e-9?14:7),color:par.map(p=>Math.abs(p.alpha-.7)<1e-9?COLORS.orange:COLORS.blue)},line:{color:COLORS.blue,width:1.5},
    customdata:par.map(p=>p.score_70_30),hovertemplate:'cost −%{x:.2f}% · CO₂ −%{y:.2f}%<br>score 70/30: %{customdata:.2f}<extra></extra>'}],
    {height:320,showlegend:false,hovermode:'closest',xaxis:{...graphTheme.xaxis,title:'Reducció del cost (%)'},yaxis:{...graphTheme.yaxis,title:'Reducció de CO₂ (%)'}},320);
  const caps=[...new Set(s.sizing.map(x=>x.capacity_kwh))],pows=[...new Set(s.sizing.map(x=>x.power_kw))];
  const z=pows.map(pw=>caps.map(c=>s.sizing.find(x=>x.capacity_kwh===c&&x.power_kw===pw)?.score));
  plot('studies-sizing',[{type:'heatmap',x:caps.map(c=>ca.format(c)+' kWh'),y:pows.map(p=>p+' kW'),z,colorscale:[[0,'#f3efe6'],[.5,'#e2a37f'],[1,'#8a3f22']],
    text:z.map(r=>r.map(v=>eu(v,2))),texttemplate:'%{text}',textfont:{size:11},colorbar:{title:'score',thickness:10},hovertemplate:'%{x} · %{y}<br>score %{z:.2f}<extra></extra>',xgap:2,ygap:2}],
    {height:320,hovermode:'closest',shapes:[{type:'rect',xref:'x',yref:'y',x0:caps.indexOf(2000)-.5,x1:caps.indexOf(2000)+.5,y0:pows.indexOf(500)-.5,y1:pows.indexOf(500)+.5,line:{color:COLORS.ink,width:2.5}}],
     annotations:[{x:caps.indexOf(2000),y:pows.indexOf(500)+.62,text:'bateria actual',showarrow:false,font:{size:10,color:COLORS.ink}}],
     xaxis:{...graphTheme.xaxis,title:'Capacitat'},yaxis:{...graphTheme.yaxis,title:'Potència'}},320);
  // Calendar: one column per ISO week, one row per weekday.
  const cal=s.calendar; if(!cal.length)return;
  const start=new Date(cal[0].date+'T12:00:00'); const monday=new Date(start); monday.setDate(start.getDate()-((start.getDay()+6)%7));
  const nW=Math.ceil((new Date(cal[cal.length-1].date+'T12:00:00')-monday)/864e5/7)+1;
  const Z=[...Array(7)].map(()=>Array(nW).fill(null)),T=[...Array(7)].map(()=>Array(nW).fill(''));
  const xs=[...Array(nW).keys()].map(i=>{const d=new Date(monday);d.setDate(monday.getDate()+7*i);return d.toISOString().slice(0,10);});
  cal.forEach(c=>{const d=new Date(c.date+'T12:00:00');const w=Math.floor((d-monday)/864e5/7);const wd=(d.getDay()+6)%7;Z[wd][w]=c.score;T[wd][w]=c.date+' · '+eu(c.score,2)+' p · '+eu(c.co2_saved_kg,0)+' kg CO₂';});
  plot('studies-calendar',[{type:'heatmap',z:Z,x:xs,y:['Dl','Dt','Dc','Dj','Dv','Ds','Dg'],text:T,hovertemplate:'%{text}<extra></extra>',colorscale:[[0,'#f3efe6'],[.4,'#e8b49b'],[.75,'#d77a50'],[1,'#743a26']],xgap:2,ygap:2,colorbar:{title:'score/dia',thickness:10}}],
    {height:300,hovermode:'closest',xaxis:{...graphTheme.xaxis,type:'date',title:''},yaxis:{...graphTheme.yaxis,autorange:'reversed'}},300);
  $('studies-corr').textContent=s.calendar_spread_corr!=null?'Correlació entre el valor diari i la dispersió del senyal 70/30 del dia: r = '+eu(s.calendar_spread_corr,2)+'.':'';
}

/* ---------- Decision Board (persistent, reproducible) ---------- */
const BOARD_KEY='upc-energy-board-v3';
function saveBoard(){
  try{localStorage.setItem(BOARD_KEY,JSON.stringify(state.board));}
  catch(e){toast('No hi ha espai al navegador. Exporta el board i elimina algunes figures.');}
}
function loadBoard(){
  try{const items=JSON.parse(localStorage.getItem(BOARD_KEY)||sessionStorage.getItem('upc-energy-board-v2')||'[]');state.board=Array.isArray(items)?items.filter(x=>x&&x.figure&&x.title).slice(0,12):[];}
  catch{state.board=[];}
  updateBoardCount();
}
function updateBoardCount(){
  $('board-counter').textContent=String(state.board.length);
  $('nav-board-count').textContent=state.board.length?String(state.board.length).padStart(2,'0'):'06';
  $('board-dataset-name').textContent=dataset()?.name||'Campus Nord';
}
function pinChart(chart,origin,insights=[],spec=null){
  if(state.board.length>=12){toast('Màxim 12 figures per informe. Exporta o elimina una targeta.');return;}
  const copy={id:'fig-'+Date.now()+'-'+Math.random().toString(36).slice(2,7),
    title:String(chart.title||'Visualització'),source:origin,dataset:dataset()?.name||state.dataset,dataset_id:state.dataset,
    insights:(Array.isArray(insights)?insights:[]).slice(0,5).map(String),spec,
    created:new Date().toLocaleString('ca-ES'),figure:chart.figure};
  state.board.push(copy);saveBoard();updateBoardCount();
  toast('Figura fixada al Decision Board');
  if(state.view==='board')renderBoard();
}
function pinGraph(id){
  const figure=figureOf(id); if(!figure){toast('Encara no hi ha gràfic per fixar.');return;}
  const panel=$(id).closest('.panel'); const title=panel?.querySelector('h3')?.textContent||'Agent Arena';
  const insights=[];
  if(id.startsWith('arena')&&state.arena){
    const pols=state.arena.policies.filter(p=>p.key!=='idle');
    pols.forEach(p=>insights.push(p.label+': '+eu(p.pooled.score,2)+' p ('+p.pooled.infractions+' infraccions)'));
    insights.splice(4);insights.push('Backtest a starter_kit/environment.py · '+state.arena.weeks.length+' setmanes pseudo-test.');
  }else if(state.studies){insights.push(state.studies.note);}
  pinChart({title,figure},'Agent Arena',insights,{type:'arena',view:id});
}
function pinLab(){
  const result=state.lastLab;
  const figure=figureOf('lab-dispatch-chart');
  if(!result||!figure){toast('Executa primer una simulació.');return;}
  const m=result.response.manual,o=result.response.optimal,q=result.req;
  pinChart({title:'Comparació de bateria · '+q.date,figure},'Decision Lab',[
    'Dia '+q.date+' · càrrega '+windowLabel(q.charge_start,q.charge_end)+' · descàrrega '+windowLabel(q.discharge_start,q.discharge_end)+' · '+q.power_kw+' kW',
    'Score '+eu(m.score,2)+' p vs òptim '+eu(o.score,2)+' p (captura '+eu(result.response.capture_pct,0)+'%)',
    'Estalvi: '+eu(m.cost_saved_eur,2)+' € · '+eu(m.co2_saved_kg,1)+' kg CO₂ · '+m.infractions+' infraccions',
    'Física i fórmula de environment.py; cicle diari tancat (SOC final ≥ 1.000 kWh).'],{type:'lab',req:q});
}
function moveBoard(index,delta){
  const target=index+delta;if(target<0||target>=state.board.length)return;
  const [item]=state.board.splice(index,1);state.board.splice(target,0,item);
  saveBoard();renderBoard();
}
function miniButton(text,tip,handler){const b=el('button','board-card-action',text);b.title=tip;b.addEventListener('click',handler);return b;}
async function rerunCard(item,holder){
  try{
    if(item.spec?.type!=='chart')return;
    const x=await post('/api/chart',item.spec.req);item.figure=x.charts[0].figure;
    saveBoard();plotFromJson(holder,item.figure);toast('Figura recalculada al servidor amb la mateixa especificació.');
  }catch(e){toast('No s’ha pogut recalcular: '+e.message);}
}
function renderBoard(){
  updateBoardCount();const empty=$('board-empty');const grid=$('board-grid');
  empty.hidden=state.board.length>0;grid.replaceChildren();
  state.board.forEach((item,index)=>{
    const card=el('article','board-card');card.draggable=true;
    const head=el('div','board-card-head');
    const num=el('span','board-card-number',hh(index+1)+' / '+item.source);
    const tools=el('div','board-card-actions');
    tools.append(miniButton('↑','Pujar',()=>moveBoard(index,-1)),miniButton('↓','Baixar',()=>moveBoard(index,1)));
    tools.append(miniButton('PNG','Exportar la figura',()=>{const g=card.querySelector('.js-plotly-plot');if(g)Plotly.downloadImage(g,{format:'png',width:1400,height:820,filename:'upc-evidence-'+(index+1)});}));
    tools.append(miniButton('×','Eliminar del board',()=>{state.board.splice(index,1);saveBoard();renderBoard();}));
    head.append(num,tools);card.append(head);
    const title=el('input','board-card-title');title.value=item.title;title.title='Pots editar aquest títol';title.setAttribute('aria-label','Títol de la figura');title.maxLength=140;
    title.addEventListener('change',()=>{item.title=title.value.slice(0,140)||item.title;saveBoard();});card.append(title);
    const meta=el('div','board-card-meta',item.dataset+' · '+item.created+(item.spec?.type==='chart'||item.spec?.type==='lab'?' · reproduïble':item.spec?.type==='question'?' · pregunta guardada':''));
    if(item.spec?.type==='chart'){const rr=el('button','board-card-action inline','↻ recalcular');rr.addEventListener('click',()=>rerunCard(item,holder));meta.append(rr);}
    card.append(meta);
    const holder=el('div','board-card-figure');card.append(holder);
    const evidence=el('div','board-card-insights');evidence.append(el('strong',null,'EVIDÈNCIA / INTERPRETACIÓ'));
    (item.insights?.length?item.insights:['Visualització calculada a partir de les dades seleccionades. Afegeix una interpretació documentada.']).forEach(v=>evidence.append(el('p',null,v)));
    const note=el('textarea','board-note');note.rows=2;note.maxLength=650;note.placeholder='Quina decisió justifica aquesta evidència? Escriu-hi la vostra conclusió.';note.value=item.note||'';
    const notePrint=el('p','board-note-print',item.note?'Conclusió de l’equip: '+item.note:'');
    note.addEventListener('change',()=>{item.note=note.value.slice(0,650);notePrint.textContent=item.note?'Conclusió de l’equip: '+item.note:'';saveBoard();});
    card.append(evidence,note,notePrint);
    card.addEventListener('dragstart',e=>{e.dataTransfer.setData('text/plain',String(index));card.classList.add('dragging');});
    card.addEventListener('dragend',()=>card.classList.remove('dragging'));
    card.addEventListener('dragover',e=>{e.preventDefault();});
    card.addEventListener('drop',e=>{e.preventDefault();const from=Number(e.dataTransfer.getData('text/plain'));if(Number.isInteger(from)&&from!==index&&from>=0&&from<state.board.length)moveBoard(from,index-from);});
    grid.append(card);
    if(state.view==='board')plotFromJson(holder,item.figure);
  });
}
function boardSummary(){
  if(!state.board.length){toast('Primer fixa alguna evidència.');return;}
  const box=$('board-summary-text');box.replaceChildren();
  const list=el('ol');
  state.board.forEach(item=>{
    const li=el('li');li.append(el('strong',null,item.title+'. '),document.createTextNode((item.insights||[]).slice(0,2).join(' ')));
    if(item.note)li.append(el('em',null,' Conclusió: '+item.note));
    list.append(li);
  });
  box.append(list);
  const sources=[...new Set(state.board.map(b=>b.source))];
  box.append(el('p','board-summary-foot','Fonts: '+sources.join(', ')+'. Tot el text prové de les evidències i notes fixades; no s’ha afegit cap xifra nova. Els resultats d’informació perfecta són cotes superiors, no previsions.'));
  $('board-summary-box').hidden=false;
}
function exportBoard(){
  if(!state.board.length){toast('Primer fixa algun gràfic al board.');return;}
  const payload={format:'upc-energy-decision-board-v3',exported_at:new Date().toISOString(),
    context:'Datathon FME 2026 · Side Challenge Tukey',methodology:'Dades històriques, càlcul al servidor, simulador oficial environment.py',
    cards:state.board};
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='UPC_Energy_Decision_Board.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  toast('Board exportat amb figures, especificacions i evidències');
}
async function importBoard(file){
  if(!file)return;
  try{
    const data=JSON.parse(await file.text());
    const cards=(Array.isArray(data?.cards)?data.cards:[]).filter(x=>x&&x.figure&&Array.isArray(x.figure.data)&&typeof x.title==='string');
    if(!cards.length)throw new Error('El fitxer no conté targetes vàlides');
    const room=12-state.board.length;
    cards.slice(0,room).forEach(c=>state.board.push({...c,id:'fig-'+Date.now()+'-'+Math.random().toString(36).slice(2,7),title:String(c.title).slice(0,140),
      insights:(c.insights||[]).slice(0,5).map(String),note:String(c.note||'').slice(0,650)}));
    saveBoard();renderBoard();toast(Math.min(room,cards.length)+' targetes importades');
  }catch(e){toast('No s’ha pogut importar: '+e.message);}
  $('board-import-input').value='';
}

/* ---------- Upload + bindings ---------- */
function exportFirst(containerId,filename){const g=$(containerId)?.querySelector('.js-plotly-plot');if(!g){toast('No hi ha cap gràfic per exportar.');return;}Plotly.downloadImage(g,{format:'png',width:1400,height:780,filename});}
function resetForDataset(){state.lastAi=null;state.lastReply=null;state.history=[];$('copilot-context').hidden=true;state.charts=[];state.aiCharts=[];state.labDates=[];state.radar=null;}
async function onUpload(file){if(!file)return;const fd=new FormData();fd.append('file',file);toast('Importació en curs: '+file.name);
  try{const result=await api('/api/upload',{method:'POST',body:fd});state.datasets.push(result);state.dataset=result.id;resetForDataset();buildDatasetSelect();await Promise.all([loadOverview(),loadQuality()]);setView('explore');await runExplore();toast('Dataset carregat: '+ca.format(result.rows)+' registres');}
  catch(e){toast(e.message);}
  $('upload-input').value='';
}
function bind(){
  document.querySelectorAll('[data-view]').forEach(n=>n.addEventListener('click',()=>setView(n.dataset.view)));
  document.querySelectorAll('[data-jump]').forEach(n=>n.addEventListener('click',()=>setView(n.dataset.jump)));
  document.querySelectorAll('[data-ask]').forEach(n=>n.addEventListener('click',()=>askAssistant(n.dataset.ask)));
  document.querySelectorAll('[data-pin-graph]').forEach(n=>n.addEventListener('click',()=>pinGraph(n.dataset.pinGraph)));
  $('dataset-select').addEventListener('change',async e=>{state.dataset=e.target.value;resetForDataset();fillMetrics();await Promise.all([loadOverview(),loadQuality(),loadDays()]);if(state.view==='explore')await runExplore();if(state.view==='lab')await loadRadar();if(state.view==='arena')await loadArena();});
  $('filter-month').addEventListener('change',loadOverview);$('filter-daytype').addEventListener('change',loadOverview);
  $('explore-run').addEventListener('click',runExplore);
  $('explore-kind').addEventListener('change',()=>{const k=$('explore-kind').value;$('explore-group').disabled=k==='heatmap'||k==='histogram'||k==='box'||k==='scatter';});
  $('explore-export').addEventListener('click',()=>exportFirst('explore-charts','upc-eda'));
  $('ai-export').addEventListener('click',()=>exportFirst('ai-canvas-charts','upc-ai-analysis'));
  $('trigger-upload').addEventListener('click',()=>$('upload-input').click());$('explore-upload').addEventListener('click',()=>$('upload-input').click());
  $('upload-input').addEventListener('change',e=>onUpload(e.target.files[0]));
  document.querySelectorAll('[data-prompt]').forEach(b=>b.addEventListener('click',()=>askAssistant(b.dataset.prompt)));
  $('chat-form').addEventListener('submit',e=>{e.preventDefault();const v=$('chat-input').value.trim();if(v)askAssistant(v);});
  $('chat-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();const v=$('chat-input').value.trim();if(v)askAssistant(v);}});
  $('radar-apply').addEventListener('click',()=>{
    if(!state.radar?.candidate)return;
    const c=state.radar.candidate;
    $('lab-charge-start').value=c.charge_hour;$('lab-charge-end').value=c.charge_hour;$('lab-discharge-start').value=c.discharge_hour;$('lab-discharge-end').value=c.discharge_hour;
    syncWindows();runLab();toast('Hipòtesi aplicada al simulador. Prova d’ampliar les finestres per capturar més valor.');
  });
  $('radar-month').addEventListener('change',loadRadar);
  $('lab-run').addEventListener('click',runLab);
  $('lab-pin').addEventListener('click',pinLab);
  $('chat-reset').addEventListener('click',()=>{state.lastAi=null;state.lastReply=null;state.history=[];$('copilot-context').hidden=true;$('chat-messages').replaceChildren();addChat('assistant','Conversa nova. Quina dada o decisió vols explorar?');toast('Nou context analític');});
  $('context-clear').addEventListener('click',()=>{state.lastAi=null;$('copilot-context').hidden=true;toast('Context netejat');});
  $('board-export').addEventListener('click',exportBoard);
  $('board-summary').addEventListener('click',boardSummary);
  $('board-import').addEventListener('click',()=>$('board-import-input').click());
  $('board-import-input').addEventListener('change',e=>importBoard(e.target.files[0]));
  $('board-print').addEventListener('click',()=>{setView('board');setTimeout(()=>window.print(),600);});
  $('arena-rerun').addEventListener('click',async()=>{try{await post('/api/arena/rerun',{});toast('Backtest reiniciat amb la versió actual de agent.py');loadArena();}catch(e){toast(e.message);}});
  ['lab-charge-start','lab-charge-end','lab-discharge-start','lab-discharge-end'].forEach(id=>$(id).addEventListener('input',syncWindows));
  ['lab-charge-start','lab-charge-end','lab-discharge-start','lab-discharge-end','lab-power'].forEach(id=>$(id).addEventListener('change',runLab));
  $('lab-power').addEventListener('input',e=>$('lab-power-value').textContent=e.target.value+' kW');
  $('lab-date').addEventListener('change',runLab);
  $('mobile-menu').addEventListener('click',()=>{document.querySelector('.sidebar').classList.add('open');$('mobile-backdrop').classList.add('visible');});
  $('mobile-backdrop').addEventListener('click',()=>{document.querySelector('.sidebar').classList.remove('open');$('mobile-backdrop').classList.remove('visible');});
}
bind();loadBoard();init();
