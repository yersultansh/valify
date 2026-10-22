#!/usr/bin/env python3
"""Frontend update: "Обновить данные" button with live status, renovation radio (Черновая отделка / Другое),
four optional spatial inputs (distance to the metro, schools / hospitals / kindergartens within 2 km),
and the OpenStreetMap attribution. Plain HTML / CSS / JavaScript, like the rest of the interface.

Run from the project root (the folder that contains frontend/):

    python3 apply_frontend_update.py

Safe by design: all edits are checked first (every anchor must match exactly once, otherwise NOTHING is
written), re-running is harmless, and a backup <file>.before_frontend is saved next to each changed file.
"""
import pathlib
import py_compile
import shutil
import sys

EDITS = {}


EDITS["frontend/index.html"] = [
(r'''  <p>Оценка по реальным объявлениям krisha.kz</p>
''', r'''  <p>Оценка по реальным объявлениям krisha.kz</p>
  <div class="refresh"><button id="refresh" type="button">Обновить данные</button>
    <span id="refresh-status" class="status" role="status" aria-live="polite"></span></div>
'''),
(r'''    <button id="submit" type="submit">Рассчитать стоимость</button>
''', r'''    <fieldset><legend>Состояние отделки</legend>
      <div class="radio">
        <label><input type="radio" name="renovation" value="0"> Черновая отделка</label>
        <label><input type="radio" name="renovation" value="1" checked> Другое</label>
      </div></fieldset>
    <p class="hint">Расположение (необязательно): если не заполнить, возьмём типичные значения района. Расстояния считаются по прямой.</p>
    <div class="field"><label for="dist_to_metro">До ближайшего метро, км</label>
      <input id="dist_to_metro" name="dist_to_metro" type="number" step="0.1" min="0" max="60" inputmode="decimal">
      <span class="err" data-for="dist_to_metro"></span></div>
    <div class="row">
      <div class="field"><label for="schools_within_2km">Школ в радиусе 2 км</label>
        <input id="schools_within_2km" name="schools_within_2km" type="number" step="1" min="0" inputmode="numeric">
        <span class="err" data-for="schools_within_2km"></span></div>
      <div class="field"><label for="hospitals_within_2km">Больниц в радиусе 2 км</label>
        <input id="hospitals_within_2km" name="hospitals_within_2km" type="number" step="1" min="0" inputmode="numeric">
        <span class="err" data-for="hospitals_within_2km"></span></div>
    </div>
    <div class="field"><label for="kindergartens_within_2km">Детских садов в радиусе 2 км</label>
      <input id="kindergartens_within_2km" name="kindergartens_within_2km" type="number" step="1" min="0" inputmode="numeric">
      <span class="err" data-for="kindergartens_within_2km"></span></div>
    <button id="submit" type="submit">Рассчитать стоимость</button>
'''),
(r'''<footer class="foot">Оценка ориентировочная и не является официальной оценкой стоимости недвижимости.</footer>''',
 r'''<footer class="foot">Оценка ориентировочная и не является официальной оценкой стоимости недвижимости.<br>Расстояния и число объектов (метро, школы, больницы, детские сады) рассчитаны по данным © участников <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> (лицензия ODbL).</footer>'''),
]

EDITS["frontend/app.js"] = [
(r'''const FIELDS = ["district", "area_total", "floor", "floor_total", "house_year", "building_type"];''',
 r'''const FIELDS = ["district", "area_total", "floor", "floor_total", "house_year", "building_type",
                "dist_to_metro", "schools_within_2km", "hospitals_within_2km", "kindergartens_within_2km"];
const SPATIAL = ["dist_to_metro", "schools_within_2km", "hospitals_within_2km", "kindergartens_within_2km"];'''),
(r'''    $("floor").min = 1; $("floor_total").min = 1;
''', r'''    $("floor").min = 1; $("floor_total").min = 1;
    for (const k of Object.keys(opts.limits || {})) { $(k).min = opts.limits[k].min; $(k).max = opts.limits[k].max; }
'''),
(r'''  return {district: $("district").value, area_total: n("area_total"), floor: n("floor"),
''', r'''  const opt = (id) => ($(id).value.trim() === "" ? null : Number($(id).value));   // optional: blank -> null
  return {district: $("district").value, area_total: n("area_total"), floor: n("floor"),
'''),
(r'''          floor_total: n("floor_total"), house_year: n("house_year"), building_type: $("building_type").value};
''', r'''          floor_total: n("floor_total"), house_year: n("house_year"), building_type: $("building_type").value,
          renovation: Number(document.querySelector('input[name="renovation"]:checked').value),
          dist_to_metro: opt("dist_to_metro"), schools_within_2km: opt("schools_within_2km"),
          hospitals_within_2km: opt("hospitals_within_2km"), kindergartens_within_2km: opt("kindergartens_within_2km")};
'''),
(r'''  if (!v.building_type) e.building_type = "Выберите тип дома.";
''', r'''  if (!v.building_type) e.building_type = "Выберите тип дома.";
  for (const k of SPATIAL) {                          // optional: only checked when filled in
    const x = v[k], lim = (opts.limits || {})[k] || {min: 0, max: 1000}, whole = k !== "dist_to_metro";
    if (x !== null && (!Number.isFinite(x) || x < lim.min || x > lim.max || (whole && !Number.isInteger(x))))
      e[k] = (whole ? "Целое число" : "Число") + " от " + lim.min + " до " + lim.max + ".";
  }
'''),
(r'''loadOptions();
''', r'''loadOptions();

// ---- "Обновить данные": POST /api/refresh, then follow GET /api/refresh/status until it finishes ----
const rbtn = $("refresh"), rstat = $("refresh-status");
function rshow(msg, busy) {
  rstat.textContent = msg || "";
  rbtn.disabled = !!busy;
  rbtn.classList.toggle("busy", !!busy);
}
async function pollRefresh() {
  for (let i = 0; i < 400; i++) {                      // up to about 20 minutes
    let s = {};
    try { s = await (await fetch("/api/refresh/status")).json(); } catch (e) { /* retry on the next tick */ }
    if (s.running === false) { rshow(s.last_result ? s.last_result.message : "Готово.", false); return; }
    rshow("Идёт обновление данных, это может занять несколько минут…", true);
    await sleep(3000);
  }
  rshow("Обновление всё ещё выполняется. Проверьте позже.", false);
}
async function startRefresh() {
  rshow("Запускаем обновление…", true);
  try {
    const r = await fetch("/api/refresh", {method: "POST"});
    const data = await r.json().catch(() => ({}));
    if (r.status === 202 || r.status === 409) return pollRefresh();      // started now, or already running
    rshow(data.message || "Не удалось запустить обновление.", false);    // 403 disabled, 429 too soon, 503 training
  } catch (e) {
    rshow("Нет связи с сервером. Проверьте, что он запущен.", false);
  }
}
rbtn.addEventListener("click", startRefresh);
fetch("/api/refresh/status").then((r) => r.json()).then((s) => { if (s.running) pollRefresh(); }).catch(() => {});
'''),
]

EDITS["frontend/style.css"] = [
(r'''@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
''', r'''fieldset{border:1px solid var(--line);border-radius:6px;margin:0 0 .9rem;padding:.5rem .75rem .6rem}
legend{font-weight:600;font-size:.9rem;padding:0 .3rem}
.radio{display:flex;gap:1.25rem;flex-wrap:wrap}
.radio label{display:flex;align-items:center;gap:.4rem;font-weight:400;margin:0}
.radio input{width:auto}
.hint{margin:.1rem 0 .7rem;color:var(--muted);font-size:.82rem}
.refresh{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;margin-top:.8rem}
.refresh button{width:auto;padding:.5rem 1rem;background:transparent;color:var(--accent);border:1px solid var(--accent)}
.refresh .status{margin:0}
.refresh button.busy::after{content:"";display:inline-block;width:.8em;height:.8em;margin-left:.5rem;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:spin .8s linear infinite;vertical-align:-.1em}
@keyframes spin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.refresh button.busy::after{animation:none}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
'''),
]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    plan, problems = {}, []
    for rel, pairs in EDITS.items():
        path = root / rel
        if not path.exists():
            problems.append(f"missing file: {rel}")
            continue
        text, applied, skipped = path.read_text(encoding="utf-8"), 0, 0
        for old, new in pairs:
            if new in text and (old not in text or old in new):      # already applied
                skipped += 1
            elif text.count(old) == 1:
                text, applied = text.replace(old, new), applied + 1
            else:
                problems.append(f"{rel}: expected exactly 1 match, found {text.count(old)} for: {old.strip()[:70]!r}")
        plan[rel] = (text, applied, skipped)
    if problems:
        print("NOTHING WAS CHANGED. Problems found:")
        for p in problems:
            print("  -", p)
        print("The files differ from the versions this update was made for; send me this output.")
        return 1
    changed = 0
    for rel, (text, applied, skipped) in plan.items():
        path = root / rel
        if applied:
            shutil.copy2(path, path.with_name(path.name + ".before_frontend"))
            path.write_text(text, encoding="utf-8")
            if path.suffix == ".py":
                py_compile.compile(str(path), doraise=True)
            changed += 1
        print(f"{rel}: {applied} edits applied, {skipped} already present")
    print("Done." if changed else "Everything was already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
