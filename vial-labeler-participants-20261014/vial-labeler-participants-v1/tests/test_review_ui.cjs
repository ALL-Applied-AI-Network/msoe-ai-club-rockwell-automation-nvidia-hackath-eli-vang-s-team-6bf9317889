// Optional browser test: npm install playwright, then node tests/test_review_ui.cjs demo/index.html.
const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),os=require('os');
const assert=require('assert/strict');
const {pathToFileURL}=require('url');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'participant-ui-'));
 try{
  const page=await browser.newPage({viewport:{width:1500,height:1050}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'demo/index.html')).href);
  assert.equal(await page.locator('.svgslot').count(),60);
  assert.equal(await page.locator('.tile').count(),30);
  assert((await page.locator('#progress').innerText()).startsWith('0 / 60'));
  await page.selectOption('#classPick','no_vial');await page.locator('#confirmed').check();
  await page.locator('#nextPosition').click();await page.locator('#previousPosition').click();
  assert.equal(await page.locator('#classPick').inputValue(),'no_vial');
  assert(await page.locator('#confirmed').isChecked());
  const event=page.waitForEvent('download');await page.locator('#download').click();
  const download=await event,file=path.join(tmp,'labels.json');await download.saveAs(file);
  const payload=JSON.parse(fs.readFileSync(file));
  assert.equal(payload.results[0].slots[0].class,'no_vial');
  assert.equal(payload.results[0].slots[0].review_status,'confirmed');
  await page.reload();assert.equal(await page.locator('#classPick').inputValue(),'no_vial');
  await page.selectOption('#classPick','missing_cap');assert(!(await page.locator('#confirmed').isChecked()));
  await page.locator('#reviewFile').setInputFiles(file);
  await page.waitForFunction(()=>document.querySelector('#loadStatus').textContent.startsWith('Loaded'));
  assert.equal(await page.locator('#classPick').inputValue(),'no_vial');
  payload.results[0].slots[0].bbox[0]+=1;
  await page.locator('#reviewFile').setInputFiles({name:'bad.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(payload))});
  await page.waitForFunction(()=>document.querySelector('#loadStatus').textContent.startsWith('Load failed'));
  assert.equal(await page.locator('#classPick').inputValue(),'no_vial');
  await page.selectOption('#areaPick','scrap');assert.equal(await page.locator('.tile').count(),30);
  if(process.argv[3])await page.screenshot({path:process.argv[3],fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert(await page.locator('#classPick').isVisible());
  assert.equal(errors.length,0,errors.join('\n'));
  console.log('Browser checks passed: navigation, confirmation reset, persistence, JSON export/import, tamper rejection, responsive layout.');
 }finally{await browser.close();fs.rmSync(tmp,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1});
