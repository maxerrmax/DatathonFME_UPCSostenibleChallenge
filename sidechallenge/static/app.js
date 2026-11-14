/* UPC Energy Intelligence — plain JavaScript, no frontend build pipeline. */
'use strict';
const $ = (id) => document.getElementById(id);
const state = {dataset:'campus', datasets:[], view:'overview', overview:null, charts:[], aiCharts:[], busy:false, labDates:[], lastAi:null, lastLab:null, board:[], lastReply:null,radar:null};
const ca = new Intl.NumberFormat('ca-ES',{maximumFractionDigits:1});
const eu = (v,dec=1) => v === null || v === undefined || !Number.isFinite(Number(v)) ? '—' : new Intl.NumberFormat('ca-ES',{minimumFractionDigits:dec,maximumFractionDigits:dec}).format(v);
const safe = (s) => String(s ?? '');
const config = {responsive:true, displaylogo:false, displayModeBar:false, scrollZoom:false};
const graphTheme = {
  paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:'DM Sans, Arial, sans-serif',color:'#7d8d8e',size:11},
  margin:{l:49,r:49,t:16,b:35},
  xaxis:{showgrid:false,zeroline:false,tickfont:{size:10},linecolor:'#e7eae6'},
  yaxis:{gridcolor:'#edf0ed',zeroline:false,tickfont:{size:10}},
  showlegend:true,legend:{orientation:'h',y:1.2,x:0,font:{size:10}},hovermode:'x unified',
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
function emptyChart(el,msg){el.replaceChildren();const n=document.createElement('div');n.className='empty-state';n.textContent=msg;el.append(n);}
function plot(id,data,layout={},height){
  const el=typeof id==='string'?$(id):id;
  if(!el)return;
  const l={...graphTheme,...layout}; if(height)l.height=height;
  Plotly.react(el,data,l,{...config});
}
function plotFromJson(el,figure){
  if (!window.Plotly) { emptyChart(el, 'La biblioteca de gràfics no s’ha carregat.'); return; }
  const h = Math.max(275,Math.min(390,window.innerHeight*.43));
  Plotly.react(el,figure.data,{...figure.layout,height:h,autosize:true},{...config,responsive:true})
    .then(()=>Plotly.Plots.resize(el))
    .catch(err=>{console.error('Error renderitzant Plotly:',err);emptyChart(el,'No s’ha pogut mostrar el gràfic.');});
}
function plotCards(containerId,charts){
  const container=$(containerId);container.replaceChildren();
  if(!charts?.length){emptyChart(container,'No s’han generat gràfics.');return;}
  charts.forEach((c,i)=>{
    const wrapper=document.createElement('div');wrapper.className='chart-entry';
    const heading=document.createElement('div');heading.className='chart-entry-head';
    const title=document.createElement('h4');title.className='chart-title';title.textContent=String(i+1).padStart(2,'0')+' / '+c.title;
    heading.append(title);
    const actions=document.createElement('div');actions.className='chart-entry-actions';
    const pin=document.createElement('button');pin.type='button';pin.className='chart-mini-action';pin.textContent='＋ Fixar al board';
    pin.title='Afegir aquest gràfic al Decision Board';
    pin.addEventListener('click',()=>pinChart(c,containerId==='ai-canvas-charts'?'AI Copilot':'Data Explorer',
      containerId==='ai-canvas-charts'?(state.lastReply?.insights||[]):['Gràfic configurat per l’usuari al Data Explorer.']));
    const download=document.createElement('button');download.type='button';download.className='chart-mini-action subtle';download.textContent='↧ PNG';
    download.addEventListener('click',()=>{const gd=wrapper.querySelector('.js-plotly-plot');if(gd) Plotly.downloadImage(gd,{format:'png',width:1400,height:820,filename:'upc-figure-'+(i+1)});});
    actions.append(pin,download);heading.append(actions);wrapper.append(heading);
    const holder=document.createElement('div');holder.className='chart-plot-holder';wrapper.append(holder);container.append(wrapper);
    plotFromJson(holder,c.figure);
  });
}
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o);}
function dataset(){return state.datasets.find(d=>d.id===state.dataset) || state.datasets[0];}
function setView(name){
  state.view=name;
  document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id==='view-'+name));
  document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x.dataset.view===name));
  $('breadcrumb-current').textContent={overview:'Vista general',explore:'Data explorer',copilot:'AI Copilot',lab:'Decision lab',board:'Decision board'}[name];
  document.querySelector('.sidebar').classList.remove('open');$('mobile-backdrop').classList.remove('visible');
  setTimeout(()=>{document.querySelectorAll('.view.active .js-plotly-plot').forEach(x=>Plotly.Plots.resize(x));},80);
  if(name==='lab'){if(!state.labDates.length)loadDays();else runLab();loadRadar();}
  if(name==='explore' && !state.charts.length)runExplore();
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
  $('data-name').textContent=d.name;
  $('data-source').textContent=d.source;
  $('data-rows').textContent=ca.format(d.rows);
  $('data-cols').textContent=ca.format(d.cols);
  $('data-missing').textContent=eu(d.missing_pct,1)+'%';
  const tmp=$('explore-kind'); if(!d.start){tmp.value='histogram';}
}
function buildDatasetSelect(){
  let selector=$('dataset-select');selector.replaceChildren();
  state.datasets.forEach(d=>option(selector,d.id,d.name));
  if(!state.datasets.some(d=>d.id===state.dataset))state.dataset=state.datasets[0]?.id||'campus';
  selector.value=state.dataset;fillMetrics();
}
async function init(){
  try{
    const status=await api('/api/status');state.datasets=status.datasets;
    $('mode-label').textContent=status.mode==='helmcode'?'HELMCODE CONFIGURAT':'MODE LOCAL';
    $('mode-sub').textContent=status.mode==='helmcode'?'Connexió pendent de verificar':'Consultes estadístiques sense API';
    $('ai-mode-state').textContent=status.mode==='helmcode'?'Helmcode configurat · pendent de prova':'Mode local · configura Helmcode per activar la IA';
    if(!state.datasets.length){toast('Falten les dades inicials. Importa un fitxer CSV per començar.');return;}
    buildDatasetSelect();await Promise.all([loadOverview(),loadDays()]);
  }catch(e){toast(e.message);}
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
    const p=x.profile;
    if(p.length){
      const hr=p.map(v=>v.hour);
      const demand={x:hr,y:p.map(v=>v.load_kw),name:'Consum · kW',type:'scatter',mode:'lines',fill:'tozeroy',line:{color:'#E77741',width:2.7},fillcolor:'rgba(231,123,80,.10)'};
      const solar={x:hr,y:p.map(v=>v.pv_kw),name:'Solar · kW',type:'scatter',mode:'lines',yaxis:'y2',line:{color:'#789A77',width:2.5}};
      let data=[demand,solar].filter(t=>t.y.some(v=>v!==undefined&&v!==null));
      plot('chart-overview-load',data,{xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora'},yaxis:{...graphTheme.yaxis,title:'Consum (kW)'},yaxis2:{title:'Solar (kW)',overlaying:'y',side:'right',showgrid:false},legend:{orientation:'h',x:0,y:1.18}});
      const price={x:hr,y:p.map(v=>v.precio_eur_per_kwh),name:'Preu · €/kWh',type:'scatter',mode:'lines+markers',line:{color:'#E77741',width:2.8},marker:{size:4}};
      const carb={x:hr,y:p.map(v=>v.co2_kg_per_kwh),name:'CO₂ · kg/kWh',type:'scatter',mode:'lines+markers',line:{color:'#577B9E',width:2.8},marker:{size:4},yaxis:'y2'};
      plot('chart-overview-market',[price,carb],{height:280,xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora'},yaxis:{...graphTheme.yaxis,title:'€/kWh'},yaxis2:{title:'kg CO₂/kWh',overlaying:'y',side:'right',showgrid:false},legend:{orientation:'h',x:0,y:1.18}});
    }else{emptyChart($('chart-overview-load'),'Aquest dataset no té hores identificades.');emptyChart($('chart-overview-market'),'Carrega dades amb data i hora.');}
    const box=$('overview-insights');box.replaceChildren();(x.recommended||[]).forEach((txt,i)=>{const el=document.createElement('div');el.className='insight-item';const n=document.createElement('i');n.textContent='0'+(i+1);const t=document.createElement('p');t.textContent=txt;el.append(n,t);box.append(el);});
    const monthSelect=$('filter-month');if(monthSelect.options.length===1 && x.dataset.start && x.dataset.end){
      const months=['gen.','feb.','març','abr.','maig','juny','jul.','ag.','set.','oct.','nov.','des.'];
      months.forEach((m,i)=>option(monthSelect,String(i+1),m));
    }
  }catch(e){toast(e.message);}
}
async function runExplore(){
  const m1=$('explore-metric').value;const m2=$('explore-second').value;
  const req={dataset_id:state.dataset,kind:$('explore-kind').value,metrics:[m1,...(m2&&m2!==m1?[m2]:[])],group:$('explore-group').value,aggregation:$('explore-agg').value};
  $('explore-run').disabled=true;$('explore-run').textContent='Generant…';
  try{const x=await api('/api/chart',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(req)});state.charts=x.charts;plotCards('explore-charts',x.charts);$('explore-title').textContent=x.charts.map(c=>c.title).join(' / ');}catch(e){toast(e.message);emptyChart($('explore-charts'),e.message);}finally{$('explore-run').disabled=false;$('explore-run').innerHTML='Generar visualització <svg><use href="#i-arrow"/></svg>';}
}
function addChat(role,lines){
  const node=document.createElement('div');node.className='message '+role;
  const label=document.createElement('div');label.className='message-label';label.textContent=role==='user'?'TU':'ENERGY COPILOT';node.append(label);
  const list=Array.isArray(lines)?lines:[lines];list.forEach(x=>{const p=document.createElement('p');p.textContent=x;node.append(p);});
  $('chat-messages').append(node);$('chat-messages').scrollTop=$('chat-messages').scrollHeight;return node;
}
async function askAssistant(question){
  if(state.busy)return;
  state.busy=true;$('chat-send').disabled=true;
  addChat('user',question);$('chat-input').value='';setView('copilot');
  const loading=addChat('assistant','Analitzant les variables i calculant visualitzacions…');
  try{
    const context=state.lastAi?.dataset===state.dataset ? state.lastAi : null;
    const payload={message:question,dataset_id:state.dataset,
      previous_query:context?.question||null,previous_plan:context?.plan||null};
    const res=await api('/api/assistant',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    loading.remove();
    const messages=res.insights.length?[...res.insights]:['No he trobat dades suficients per extreure una conclusió.'];
    if(res.mode==='context'){
      $('ai-mode-state').textContent='Context reutilitzat · càlcul verificat';
    }
    if(res.mode==='local' && res.llm_warning){
      messages.unshift('Avís: Helmcode no ha respost. S’ha fet servir el planificador local. '+res.llm_warning);
      $('ai-mode-state').textContent='Mode local de reserva · consulta els detalls';
      $('mode-label').textContent='FALLBACK LOCAL';
    }else if(res.mode==='helmcode'){
      $('ai-mode-state').textContent='Helmcode verificat · càlcul local';
      $('mode-label').textContent='HELMCODE VERIFICAT';
    }
    addChat('assistant',messages);
    state.lastReply=res;
    if(res.mode!=='physics')state.lastAi={dataset:state.dataset,question,plan:res.plan};
    $('copilot-context').hidden=!state.lastAi;
    if(state.lastAi){$('context-label').textContent='Context actiu · '+state.lastAi.plan.metrics.map(m=>dataset()?.metrics.find(x=>x.key===m)?.label||m).join(' + ');}
    state.aiCharts=res.charts;plotCards('ai-canvas-charts',res.charts);
    $('ai-canvas-title').textContent=res.charts.map(c=>c.title).join(' / ');
    $('ai-canvas-subtitle').textContent=(res.mode==='helmcode'?'Planificat amb Helmcode':res.mode==='context'?'Context anterior verificat':res.mode==='physics'?'Simulació física verificada':'Planificador local')+' · '+ca.format(res.evidence.rows)+' registres';
    $('ai-evidence').hidden=false;
    $('ai-evidence-text').replaceChildren();
    const p=document.createElement('p');p.textContent=res.note;$('ai-evidence-text').append(p);
    const p2=document.createElement('p');p2.textContent='Operació: '+res.plan.kind+' · agregació: '+res.plan.aggregation+' · variables: '+res.plan.metrics.join(', ');$('ai-evidence-text').append(p2);
  }catch(e){loading.remove();addChat('assistant','No s’ha pogut completar l’anàlisi: '+e.message);toast(e.message);}
  finally{state.busy=false;$('chat-send').disabled=false;}
}
async function loadRadar(){
  try{
    const result=await api('/api/radar?dataset_id='+encodeURIComponent(state.dataset));
    state.radar=result;
    $('radar-charge').textContent=String(result.candidate.charge_hour).padStart(2,'0')+':00';
    $('radar-discharge').textContent=String(result.candidate.discharge_hour).padStart(2,'0')+':00';
    $('radar-reliability').textContent=result.candidate.viable_day_pct==null?'—':eu(result.candidate.viable_day_pct,1)+'%';
    $('radar-note').textContent=result.disclaimer+' Mostra: '+result.candidate.days_examined+' dies complets.';
    const ticks=result.profile.map(x=>x.hour),values=result.profile.map(x=>x.signal);
    const colors=ticks.map(h=>h===result.candidate.charge_hour?'#79a18a':h===result.candidate.discharge_hour?'#db7d53':'#bfcfca');
    plot('radar-chart',[{type:'bar',x:ticks,y:values,marker:{color:colors},name:'Senyal combinat',
      hovertemplate:'%{x}:00 · índex %{y:.3f}<extra></extra>'}],
      {height:265,showlegend:false,margin:{l:45,r:17,t:18,b:38},
       xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora del dia'},
       yaxis:{...graphTheme.yaxis,title:'Índex (mitjana = 1)'}},265);
  }catch(e){$('radar-note').textContent='Radar no disponible: '+e.message;}
}
async function loadDays(){
  const select=$('lab-date');
  try{
    const res=await api('/api/days?dataset_id='+encodeURIComponent(state.dataset));state.labDates=res.dates;
    select.replaceChildren();if(!res.dates.length){option(select,'','Sense dies complets');toast(res.reason||'Aquest dataset no és apte per simular la bateria.');return;}
    // Use an early full train date; date is explicit, does not cross train/test gaps.
    res.dates.slice(0,365).forEach(d=>option(select,d,d));
    select.value=res.dates.find(d=>d.startsWith('2025-09'))||res.dates[0];
    await runLab();
  }catch(e){toast(e.message);}
}
async function runLab(){
  if(!$('lab-date').value)return;
  const req={dataset_id:state.dataset,date:$('lab-date').value,charge_hour:+$('lab-charge').value,discharge_hour:+$('lab-discharge').value,power_kw:+$('lab-power').value};
  $('lab-run').disabled=true;
  try{
    const x=await api('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(req)});
    $('lab-base-cost').textContent=eu(x.baseline.cost,0)+' €';
    $('lab-man-score').textContent=eu(x.manual.score,2)+' p';
    $('lab-opt-score').textContent=eu(x.optimal.score,2)+' p';
    $('lab-notice').textContent=x.notice;
    state.lastLab={response:x, date:req.date, chargeHour:req.charge_hour, dischargeHour:req.discharge_hour, power:req.power_kw};
    const bar=x.manual_actions.map((a,i)=>a>=0?'#7E9F83':'#E77B50');
    plot('lab-dispatch-chart',[
      {x:x.hours,y:x.manual_actions,type:'bar',name:'Acció manual · kW',marker:{color:bar},opacity:.9},
      {x:x.hours,y:x.manual_soc,type:'scatter',mode:'lines+markers',name:'SOC manual · kWh',yaxis:'y2',line:{color:'#517698',width:2.6},marker:{size:4}},
      {x:x.hours,y:x.optimal_soc,type:'scatter',mode:'lines',name:'SOC òptim · kWh',yaxis:'y2',line:{color:'#A9B4AC',width:2,dash:'dot'}},
    ],{height:350,barmode:'relative',xaxis:{...graphTheme.xaxis,dtick:2,title:'Hora del dia'},yaxis:{...graphTheme.yaxis,title:'Potència (kW)',zeroline:true,zerolinecolor:'#CCD6D1'},yaxis2:{title:'SOC (kWh)',overlaying:'y',side:'right',range:[0,2100],showgrid:false},legend:{orientation:'h',y:1.17,x:0}},350);
    const table=$('lab-result-table');table.replaceChildren();
    const dom=document.createElement('table');const thead=document.createElement('thead');const tr=document.createElement('tr');
    ['Mètrica','La teva decisió','Òptim retrospectiu'].forEach(t=>{const th=document.createElement('th');th.textContent=t;tr.append(th);});thead.append(tr);dom.append(thead);
    const rows=[['CO₂ estalviat',eu(x.manual.saving_kg,2)+' kg',eu(x.optimal.saving_kg,2)+' kg'],['Cost estalviat',eu(x.manual.saving_eur,2)+' €',eu(x.optimal.saving_eur,2)+' €'],['Reducció d’emissions',eu(x.manual.saving_co2_pct,2)+'%',eu(x.optimal.saving_co2_pct,2)+'%'],['Reducció del cost',eu(x.manual.saving_cost_pct,2)+'%',eu(x.optimal.saving_cost_pct,2)+'%'],['SOC final',eu(x.manual.soc_end_kwh,1)+' kWh',eu(x.optimal.soc_end_kwh,1)+' kWh']];
    const body=document.createElement('tbody');rows.forEach(r=>{const row=document.createElement('tr');r.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;row.append(td);});body.append(row);});dom.append(body);table.append(dom);
  }catch(e){toast(e.message);}finally{$('lab-run').disabled=false;}
}
function exportFirst(containerId,filename){const el=$(containerId)?.querySelector('.js-plotly-plot');if(!el){toast('No hi ha cap gràfic per exportar.');return;}Plotly.downloadImage(el,{format:'png',width:1400,height:780,filename});}
async function onUpload(file){if(!file)return;const fd=new FormData();fd.append('file',file);toast('Importació en curs: '+file.name);
  try{const result=await api('/api/upload',{method:'POST',body:fd});state.datasets.push(result);state.dataset=result.id;state.lastAi=null;state.lastReply=null;buildDatasetSelect();state.charts=[];state.aiCharts=[];state.labDates=[];state.radar=null;$('filter-month').replaceChildren();option($('filter-month'),'','Tots els mesos');await loadOverview();setView('explore');await runExplore();toast('Dataset carregat: '+ca.format(result.rows)+' registres');}
  catch(e){toast(e.message);}
  $('upload-input').value='';
}
function bind(){
  document.querySelectorAll('[data-view]').forEach(el=>el.addEventListener('click',()=>setView(el.dataset.view)));
  document.querySelectorAll('[data-jump]').forEach(el=>el.addEventListener('click',()=>setView(el.dataset.jump)));
  $('dataset-select').addEventListener('change',async e=>{state.dataset=e.target.value;state.lastAi=null;state.lastReply=null;$('copilot-context').hidden=true;fillMetrics();state.charts=[];state.aiCharts=[];state.labDates=[];state.radar=null;$('filter-month').replaceChildren();option($('filter-month'),'','Tots els mesos');await loadOverview();await loadDays();if(state.view==='explore')await runExplore();if(state.view==='lab')await loadRadar();});
  $('filter-month').addEventListener('change',loadOverview);$('filter-daytype').addEventListener('change',loadOverview);
  $('explore-run').addEventListener('click',runExplore);
  $('explore-kind').addEventListener('change',()=>{const k=$('explore-kind').value;const group=$('explore-group');group.disabled=k==='heatmap'||k==='histogram'||k==='box'||k==='scatter';});
  $('explore-export').addEventListener('click',()=>exportFirst('explore-charts','upc-eda'));
  $('ai-export').addEventListener('click',()=>exportFirst('ai-canvas-charts','upc-ai-analysis'));
  $('trigger-upload').addEventListener('click',()=>$('upload-input').click());$('explore-upload').addEventListener('click',()=>$('upload-input').click());
  $('upload-input').addEventListener('change',e=>onUpload(e.target.files[0]));
  document.querySelectorAll('[data-prompt]').forEach(b=>b.addEventListener('click',()=>askAssistant(b.dataset.prompt)));
  $('chat-form').addEventListener('submit',e=>{e.preventDefault();const v=$('chat-input').value.trim();if(v)askAssistant(v);});
  $('chat-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();const v=$('chat-input').value.trim();if(v)askAssistant(v);}});
  $('radar-apply').addEventListener('click',()=>{
    if(!state.radar?.candidate)return;
    $('lab-charge').value=state.radar.candidate.charge_hour;
    $('lab-discharge').value=state.radar.candidate.discharge_hour;
    ['charge','discharge'].forEach(n=>{$('lab-'+n+'-value').textContent=String($('lab-'+n).value).padStart(2,'0')+':00';});
    runLab();toast('Hipòtesi aplicada al simulador. Consulta el resultat físic.');
  });
  $('lab-run').addEventListener('click',runLab);
  $('lab-pin').addEventListener('click',pinLab);
  $('chat-reset').addEventListener('click',()=>{state.lastAi=null;state.lastReply=null;$('copilot-context').hidden=true;$('chat-messages').replaceChildren();addChat('assistant','Conversa nova. Quina dada o decisió vols explorar?');toast('Nou context analític');});
  $('context-clear').addEventListener('click',()=>{state.lastAi=null;$('copilot-context').hidden=true;toast('Context netejat');});
  $('board-export').addEventListener('click',exportBoard);
  $('board-print').addEventListener('click',()=>{setView('board');setTimeout(()=>window.print(),320);});
  ['charge','discharge'].forEach(n=>{$('lab-'+n).addEventListener('input',e=>{$('lab-'+n+'-value').textContent=String(e.target.value).padStart(2,'0')+':00';});});
  $('lab-power').addEventListener('input',e=>$('lab-power-value').textContent=e.target.value+' kW');
  $('lab-date').addEventListener('change',runLab);
  $('mobile-menu').addEventListener('click',()=>{document.querySelector('.sidebar').classList.add('open');$('mobile-backdrop').classList.add('visible');});
  $('mobile-backdrop').addEventListener('click',()=>{document.querySelector('.sidebar').classList.remove('open');$('mobile-backdrop').classList.remove('visible');});
}
// Editorial Decision Board: figures pinned from AI, Explorer, or simulations.
const BOARD_KEY='upc-energy-board-v2';
function saveBoard(){
  try{sessionStorage.setItem(BOARD_KEY,JSON.stringify(state.board));}
  catch(e){toast('No hi ha espai al navegador. Exporta el board i elimina algunes figures.');}
}
function loadBoard(){
  try{const items=JSON.parse(sessionStorage.getItem(BOARD_KEY)||'[]');state.board=Array.isArray(items)?items.filter(x=>x&&x.figure&&x.title).slice(0,10):[];}
  catch{state.board=[];}
  updateBoardCount();
}
function updateBoardCount(){
  $('board-counter').textContent=String(state.board.length);
  $('nav-board-count').textContent=state.board.length?String(state.board.length).padStart(2,'0'):'05';
  $('board-dataset-name').textContent=dataset()?.name||'Campus Nord';
}
function pinChart(chart,origin,insights=[],extra={}){
  if(state.board.length>=10){toast('Màxim 10 figures per informe. Exporta o elimina una targeta.');return;}
  const copy={id:'fig-'+Date.now()+'-'+Math.random().toString(36).slice(2,7),
    title:String(chart.title||'Visualització'),source:origin,dataset:dataset()?.name||state.dataset,
    insights:(Array.isArray(insights)?insights:[]).slice(0,4).map(String),
    created:new Date().toLocaleString('ca-ES'),figure:chart.figure, ...extra};
  state.board.push(copy);saveBoard();updateBoardCount();
  toast('Figura fixada al Decision Board');
  if(state.view==='board')renderBoard();
}
function pinLab(){
  const result=state.lastLab;
  const graph=$('lab-dispatch-chart');
  if(!result||!graph?.data?.length){toast('Executa primer una simulació.');return;}
  const m=result.response.manual,o=result.response.optimal;
  const figure={data:JSON.parse(JSON.stringify(graph.data)),layout:JSON.parse(JSON.stringify(graph.layout))};
  pinChart({title:'Comparació de bateria · '+result.date,figure},'Decision Lab',[
    'Dia: '+result.date+' · càrrega '+result.chargeHour+':00 · descàrrega '+result.dischargeHour+':00 · '+result.power+' kW',
    'Score manual: '+m.score+' · òptim retrospectiu: '+o.score,
    'Estalvi manual: '+m.saving_eur+' € · '+m.saving_kg+' kg CO₂',
    'Simulació retrospectiva, no score oficial; SOC inicial i final del dia fixats al 50%.']);
}
function moveBoard(index,delta){
  const target=index+delta;if(target<0||target>=state.board.length)return;
  const [item]=state.board.splice(index,1);state.board.splice(target,0,item);
  saveBoard();renderBoard();
}
function miniButton(text,tip,handler){const b=document.createElement('button');b.className='board-card-action';b.textContent=text;b.title=tip;b.addEventListener('click',handler);return b;}
function renderBoard(){
  updateBoardCount();const empty=$('board-empty');const grid=$('board-grid');
  empty.hidden=state.board.length>0;grid.replaceChildren();
  state.board.forEach((item,index)=>{
    const card=document.createElement('article');card.className='board-card';card.draggable=true;
    const head=document.createElement('div');head.className='board-card-head';
    const num=document.createElement('span');num.className='board-card-number';num.textContent=String(index+1).padStart(2,'0')+' / '+item.source;
    const tools=document.createElement('div');tools.className='board-card-actions';
    tools.append(miniButton('↑','Pujar',()=>moveBoard(index,-1)),miniButton('↓','Baixar',()=>moveBoard(index,1)));
    tools.append(miniButton('PNG','Exportar la figura',()=>{const el=card.querySelector('.js-plotly-plot');if(el)Plotly.downloadImage(el,{format:'png',width:1400,height:820,filename:'upc-evidence-'+(index+1)}); }));
    tools.append(miniButton('×','Eliminar del board',()=>{state.board.splice(index,1);saveBoard();renderBoard();}));
    head.append(num,tools);card.append(head);
    const title=document.createElement('input');title.className='board-card-title';title.value=item.title;title.title='Pots editar aquest títol';title.setAttribute('aria-label','Títol de la figura');title.maxLength=140;
    title.addEventListener('change',()=>{item.title=title.value.slice(0,140)||item.title;saveBoard();});card.append(title);
    const meta=document.createElement('div');meta.className='board-card-meta';meta.textContent=item.dataset+' · '+item.created;card.append(meta);
    const holder=document.createElement('div');holder.className='board-card-figure';card.append(holder);
    const evidence=document.createElement('div');evidence.className='board-card-insights';
    const kicker=document.createElement('strong');kicker.textContent='EVIDÈNCIA / INTERPRETACIÓ';evidence.append(kicker);
    if(item.insights?.length){item.insights.forEach(value=>{const p=document.createElement('p');p.textContent=value;evidence.append(p);});}
    else{const p=document.createElement('p');p.textContent='Visualització calculada a partir de les dades seleccionades. Afegeix una interpretació documentada.';evidence.append(p);}
    const note=document.createElement('textarea');note.className='board-note';note.rows=2;note.maxLength=650;note.placeholder='Quina decisió justifica aquesta evidència? Escriu-hi la vostra conclusió.';note.value=item.note||'';
    note.addEventListener('change',()=>{item.note=note.value.slice(0,650);saveBoard();});
    card.append(evidence,note);
    card.addEventListener('dragstart',e=>{e.dataTransfer.setData('text/plain',String(index));card.classList.add('dragging');});
    card.addEventListener('dragend',()=>card.classList.remove('dragging'));
    card.addEventListener('dragover',e=>{e.preventDefault();});
    card.addEventListener('drop',e=>{e.preventDefault();const from=Number(e.dataTransfer.getData('text/plain'));if(Number.isInteger(from)&&from!==index&&from>=0&&from<state.board.length)moveBoard(from,index-from);});
    grid.append(card);
    if(state.view==='board')plotFromJson(holder,item.figure);
  });
}
function exportBoard(){
  if(!state.board.length){toast('Primer fixa algun gràfic al board.');return;}
  const payload={format:'upc-energy-decision-board-v2',exported_at:new Date().toISOString(),
    context:'Datathon FME 2026 · Side Challenge Tukey',methodology:'Dades històriques, càlcul local, simulacions retrospectiva',
    cards:state.board};
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='UPC_Energy_Decision_Board.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  toast('Board exportat amb les figures i les evidències');
}
bind();loadBoard();init();
