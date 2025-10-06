Aufgabenliste: Tensor → Quantenzustand (End-to-End)
1) Tensor exakt spezifizieren

1.1 Spezifikations-Checkliste (ausfüllen)
- Zweck/Use-Case: Wofür wird T verwendet (Encoding, Training, Inferenz)?
- Datentyp/Präzision: {float32 | float64 | int8 | uint8 | complex64 …}
- Wertebereich & Einheiten: min/max, Skala, Offset, erlaubte Sonderwerte (NaN/Inf?)
- Form/Shape: {d | H×W×C | …} und feste Größen vs. variabel; Batch separat halten
- Ordnung/Indexierung: {row-major | col-major}; klare i ↔ Koordinate-Abbildung
- Reell/komplex: Reellwerte, oder getrennte Real/Imag/Phase-Kodierung
- Normalisierung: mean/std oder L2-Norm (bes. für Amplituden-Encoding Pflicht)
- Quantisierung/Clipping: Ja/Nein; Bitbreite, Schwellen, Rundungsmodus
- Padding-Regeln: auf 2^n für Amplituden; Pad-Wert, Maskenführung
- Ziel (für Schritt 5): {|ψ⟩ Amplituden/Phasen | Parameter Θ (Winkel)}
- Observablen-Vorplanung: welche Messoperatoren sind relevant (Z, ZZ, Pauli-Strings)?
- Metadaten/Versionierung: Spec-ID, Datum, Autor, Datensatz-Version

1.2 Akzeptanzkriterien (automatisierbar)
- Shape stimmt exakt mit Spezifikation überein
- dtype stimmt; keine impliziten Up/Downcasts beim Laden
- Wertebereich eingehalten; optional assert min/max
- Keine NaN/Inf nach Vorverarbeitung
- Für Amplituden-Encoding: ||x||2 = 1 ± 1e-7 (nach Padding & Normierung)
- Für Angle/Phase: Winkel liegen innerhalb definierter Grenzen (z. B. [−π, π])
- Indexierungsrunde: Stichprobe i ↔ Koordinate ergibt erwartete Position

1.3 Vorlage (zum Ausfüllen)
```yaml
tensor_spec:
	id: qnn.tensor.v1
	purpose: <z. B. Input-Features für Angle-Encoding>
	dtype: float32
	value_range:
		min: -1.0
		max: 1.0
		units: normalized
		allow_nan_inf: false
	shape:
		layout: HxWxC
		H: <int>
		W: <int>
		C: <int>
		batch_dimension: separate
	indexing:
		order: row-major
		mapping_note: "i ↔ (h,w,c) with i = ((h*W)+w)*C + c"
	complex:
		representation: real
	preprocessing:
		normalize: {type: none | l2 | zscore, params: {}}
		clip: {enabled: false, min: null, max: null}
		quantize: {enabled: false, bits: null, mode: null}
		pad:
			to_power_of_two: {enabled: false, axis: flattened, pad_value: 0.0}
	encoding_goal:
		target: angle  # angle | amplitude | phase | basis
		angles:
			alpha: <float>  # θ = α x, keep within [−π, π]
		amplitude:
			require_l2_norm: true
		phase:
			beta: <float>
	observables_plan:
		- name: Z_chain
			desc: Z⊗…⊗Z on data qubits
	versioning:
		date: 2025-08-26
		author: <name>
		dataset_version: <id>
```

1.4 Beispiele (Richtwerte)
- Beispiel A (Bild 28×28×1, Angle-Encoding): dtype=float32, Range [0,1] → rescale zu [−1,1]; α=π so dass θ ∈ [−π, π]; row-major Flatten, i = h·W·C + w·C + c.
- Beispiel B (Vektor d=1000, Amplituden-Encoding): pad auf d’=1024, n=10 Qubits; L2-Normierung strikt, pad mit 0, Maske m für erste 1000 Indizes speichern.

Form: 
𝑇
∈
𝑅
𝑑
T∈R
d
 oder 
𝑅
𝐻
×
𝑊
×
𝐶
R
H×W×C
 etc.

Ziel: Zustand 
∣
𝜓
⟩
∣ψ⟩ (Amplitude/Phase) oder Parameter 
Θ
Θ (Winkel).

Genauigkeit/Skalierung: Welche Wertebereiche sind erlaubt? (Clips, Quantisierung)

Achtung: Wenn du später messen willst, plane welche Observablen (z. B. Z, ZZ, Pauli-Strings) du brauchst. Das beeinflusst das Circuit-Design.

2) Vorverarbeitung (immer)

2.1 Flatten & Ordnung festlegen

Lege eine stabile Indexierung fest (z. B. row-major für Bilder).

Dokumentiere: Index 
𝑖
↔
i↔ Tensor-Koordinate.

2.2 Skalierung

Angle-Encoding: skaliere 
𝑥
x in sinnvollen Bereich, z. B. 
[
−
1
,
1
]
[−1,1] und nutze 
𝜃
=
𝛼
𝑥
θ=αx mit 
𝛼
α so, dass Winkel in 
[
−
𝜋
,
𝜋
]
[−π,π] bleiben.

Amplitude-Encoding: normiere 
𝑥
x strikt: 
𝑥
~
=
𝑥
/
∥
𝑥
∥
2
x
~
=x/∥x∥
2
	​

.

Phase-Encoding: mappe 
𝑥
x auf 
𝜙
=
𝛽
𝑥
ϕ=βx (Range wählen, Aliasing vermeiden).

2.3 Padding auf Potenz von 2 (für Amplituden)

𝑑
’
=
2
⌈
log
⁡
2
𝑑
⌉
d’=2
⌈log
2
	​

d⌉
; padde mit Nullen bis Länge 
𝑑
’
d’.

Notiere Masken/Indizes, damit du später Auswertung korrekt de-paddest.

Achtung (häufige Fehler):

Ungenau dokumentierte Index-Ordnung → falsches Mapping.

Fehlende L2-Norm vor Amplituden-Encoding → falscher Zustand.

Zu aggressive Clipping/Quantisierung → Informationsverlust.

3) Qubit-Budget & Layout planen

Amplitude-Encoding: 
𝑛
=
⌈
log
⁡
2
𝑑
’
⌉
n=⌈log
2
	​

d’⌉ Qubits.

Angle-Encoding: mindestens 
𝑞
q Qubits (typisch 
𝑞
q = Feature-Slots pro Upload). Wenn 
𝑑
≫
𝑞
d≫q, plane L Re-Upload-Schichten → insgesamt 
𝐿
⋅
𝑞
≥
𝑑
L⋅q≥d (bei einfacher 1:1-Zuordnung; bei Feature-Maps mit Entanglement reduziert sich L).

Kopplung/Connectivity: plane Entanglement-Gatter kompatibel mit Hardware-Topologie (line, heavy-hex, etc.).

Achtung: Mehr Qubits ≠ besser. Rauschen steigt, transpilation wird tiefer.

4) Encoding-Entscheidung treffen (Checkliste)

Wenn du maximale Qubit-Effizienz willst und kannst tiefe Prep tolerieren → Amplitude.

Wenn du Stabilität/Einfachheit willst → Angle (+ Re-Upload + Entanglement-Blöcke).

Wenn Wert nur Index ist → Basis.

Optional: Kombis (z. B. Angle + kontrollierte ZZ-Entangler als Feature-Map).

5) State-Preparation / Parameter-Zuordnung bauen
5A) Angle-Encoding (empfohlen für NISQ)

Schritt A1: Teile 
𝑥
x in Chunks der Größe 
𝑞
q (Anzahl Daten-Qubits pro Layer).

Schritt A2: Für jeden Chunk 
𝑐
c:

wende auf jedes Qubit 
𝑗
j eine Rotation an, z. B. 
Ry
(
𝛼
⋅
𝑐
𝑗
)
Ry(α⋅c
j
	​

).

füge Entanglement (z. B. CZ-Kette oder CNOT-Ring) ein → erfasst Korrelationen.

Schritt A3: Wiederhole für L Re-Upload-Layer (neue Chunks oder gleiche Features erneut, je nach Feature-Map-Design).

Validierung: simuliere ideal (Statevector) und prüfe Erwartungswerte für Test-Observablen.

Achtung:

Zu große 
𝛼
α → Winkel-Wrap/Periodizität.

Zu tiefe Entangler → barren plateaus (Gradienten ~0). Gegenmittel: flache, problem-inspirierte Feature-Maps, layer-weise Training, SPSA.

5B) Amplitude-Encoding (qubit-effizient, aber aufwendig)

Schritt B1: 
𝑥
~
=
𝑥
/
∥
𝑥
∥
x
~
=x/∥x∥, pad auf 
𝑑
’
=
2
𝑛
d’=2
n
.

Schritt B2: Nutze einen State-Prep-Algorithmus (z. B. Möttönen-Methode / Grover-Rudolph-Variante), der rekursiv Rotationen auf kontrollierten Achsen setzt, um 
∣
𝜓
⟩
=
∑
𝑥
~
𝑖
∣
𝑖
⟩
∣ψ⟩=∑
x
~
i
	​

∣i⟩ vorzubereiten.

Schritt B3: Prüfe per Stichproben-Tomographie (oder Overlap-Schätzung), ob die Amplituden stimmen (inner product zu Referenz).

Komplexität (wichtig): allgemeines Amplituden-Laden ist O(d) an Rotationen/Kontrollen; auf NISQ schnell tief/rausch-empfindlich.

Achtung:

Genaues L2-Norming und Zahlenbereich (float → Winkelpräzision).

Signen: Komplexe/negative Werte benötigen Phasen-Korrekturen (z. B. zusätzliche Z/Phase-Gates).

Noise ruiniert feine Amplituden → ggf. erst komprimieren (PCA/TT-SVD) und dann laden.

5C) Phase-Encoding (optional)

Setze 
𝜙
𝑖
=
𝛽
𝑥
𝑖
ϕ
i
	​

=βx
i
	​

 und appliziere kontrollierte Phasen auf 
∣
𝑖
⟩
∣i⟩ (oder direkte Rz auf adressierten Qubits bei Feature-Maps).

Gut, wenn Interferenz-muster wichtiger sind als Amplituden.

6) Spezialfälle: Bilder & Convolution-Kerne

Bilder (H×W×C):

Flatten mit Struktur (erst Pixelposition → dann Kanal), oder nutze zwei Register: Positions-Register (für |i⟩) + Wert-Register (Winkel/Amplitude).

Downsampling/Kompression (PCA, DCT, Patch-Averaging) vor Amplituden-Encoding.

Kerne (k×k×C): kleine Tensoren eignen sich gut für Angle-Encoding als Parameter in wenigen Qubits; entangling-Pattern kann Nachbarschaft im Kernel widerspiegeln.

Achtung: Erhalte lokale Struktur—vermeide wahlloses Flatten, wenn du mit Entanglement Nachbarschaften modellieren willst.

7) Mess-Design (wie prüfst du „richtig geladen“)

Definiere Observablen 
𝑂
O (z. B. Z⊗Z, X⊗I…); Ziel ist 
⟨
𝜓
∣
𝑂
∣
𝜓
⟩
⟨ψ∣O∣ψ⟩.

Für Amplituden-Encoding kannst du punktuell Amplitude/Probability-Checks machen (zähle Häufigkeiten der |i⟩-Outcomes).

Für Angle-Encoding vergleiche Feature-zu-Winkel-Mapping über kontrollierte Test-Circuits (z. B. identische Layer klassisch simulieren und Erwartungswerte matchen).

Achtung: Shot-Noise (endliche Mess-wiederholungen). Nutze genügend shots und readout-Mitigation (einfaches Kalibrieren).

8) Korrektheits- und Robustheits-Checks

Roundtrip-Test (soweit möglich): klassisch → encode → (inverse Prep) → klassisch rekonstruieren; prüfe MSE.

Skalierungs-Sweep: teste verschiedene 
𝛼
,
𝛽
α,β (Winkelskalen) und Layers 
𝐿
L.

Noise-Sweep: simuliere mit Rauschmodell; prüfe, ab wann Metriken kippen.

A/B Feature-Maps: vergleiche einfache Ry-Map vs. z. B. ZZ-FeatureMap (zusätzliche Wechselwirkungsterme).

9) Performance-Tuning (wenn’s groß wird)

Kompression vor Encoding: PCA/Random-Features/TT-SVD auf 
𝑥
x, dann Angle-Encoding.

Tensor-Netzwerk-Tricks: Tree/Tensor-Train-basierte State-Prep reduziert Tiefe für strukturierte Daten.

Data-Reuploading statt große n: gleiche Qubits, mehrere Re-Upload-Layers (gute Praxis auf NISQ).

Sparsame Amplituden: wenn 
𝑥
x spärlich ist, nutze selektive State-Prep (nur relevante |i⟩-Äste).

Block-Encoding (fortgeschritten): Matrix 
𝐴
A als Teil eines Unitärs einbetten; praktikabel vor allem für sparsame oder low-rank 
𝐴
A. Dicht & groß ist teuer.

Achtung (harte Grenzen): Allgemeines, schnelles Laden beliebiger großer Tensoren ist offen/teuer; „schnelles QRAM“ ist nicht praktisch verfügbar.

10) Mini-Vorlagen (pseudocode-artig)

Angle-Encoding mit Re-Upload (Skizze):

# Inputs: x (len d), q qubits per layer, L layers, scale alpha
chunks = split(x, size=q, pad_with=0)
for layer in range(L):
    for j in range(q):
        Ry(alpha * chunks[layer][j]) on qubit j
    Entangle(qubits)  # e.g., CZ chain


Amplitude-Encoding (Skizze):

# Input: x (len d), pad to d' = 2^n, normalize
x = pad_to_power_of_two(x)
x = x / l2norm(x)
StatePrep_Mottonen(x)  # builds recursive controlled rotations


Phase-Encoding (Skizze):

for i in addressed_indices:
    apply controlled-Phase( beta * x[i] ) to basis |i>
# or Rz(beta * feature) on selected qubits in a feature-map

11) Typische Fallschritte (kurz & konkret)

Keine saubere Normierung bei Amplitude-Encoding → völlig falscher Zustand.

Zu tiefe Circuits → barren plateaus; starte flach, Layer-weise erweitern.

Falsche Feature-Skalen → Winkel-Sättigung/Periodizität.

Zu wenige Shots → verrauschte Metriken, falsche Schlüsse.

Ignorierte Hardware-Topologie → Transpiler bläst Tiefe auf.

Strukturverlust beim Flatten → schlechtere Lernleistung; lieber strukturierte Feature-Maps.

12) Entscheidungsbaum (kurz)

d groß, Qubits knapp, Rauschen okay? → Angle + Re-Upload + Entangler.

d groß, du brauchst volle Vektorinhalte kompakt? → Amplitude, aber vorher komprimieren.

Diskrete Indizes/Klassen? → Basis (oder Angle auf kleinen One-Hot-Projektionen).

Interferenz-lastig? → Phase (oder kombinierte Feature-Map).

## Fortschritt (26.08.2025)

- [x] Spezifikation — `specs/tensor_spec.yaml`, `scripts/smoke_validate_spec.py`
- [x] Vorverarbeitung — `qnn/preprocess.py`, Demos/Tests: `scripts/demo_preprocess.py`, `tests/test_preprocess.py`
- [x] Planung — Angle q,L in `qnn/planning.py`
- [x] Encoding-Pfade — `qnn/circuits.py`; Demos: `scripts/demo_angle_sim.py`, `scripts/demo_phase_sim.py`, `scripts/demo_amplitude_sim.py`
- [x] Spezialfälle/Demos — `scripts/demo_image_angle.py`, `scripts/demo_kernel_grid.py`
- [x] Mess-Design — `scripts/demo_measurements.py`
- [x] Robustheit/Noise — `scripts/demo_noise_angle.py`
- [x] Transpilation/Backend — `qnn/transpile.py`, `scripts/demo_transpile_angle.py`
- [x] Sweeps & Plots — `scripts/sweep_alpha_L.py`, `scripts/sweep_noise.py`, `scripts/plot_sweeps.py`, Artefakte: `reports/*.png`
- [x] VQA & Classifier — `qnn/vqa.py`, Demos: `scripts/demo_vqa_spsa.py`, `scripts/demo_reupload_classifier.py`
- [x] Training — `scripts/train_reupload_classifier.py` (Logs → `reports/train_reupload_classifier.json`, Params → `reports/reupload_params.json`)
- [x] CLI/UX & Config — Konsolenbefehle `qnn-train`/`qnn-predict`; `qnn/config.py`, `configs/example.yaml`, `qnn/models.py`, `scripts/predict_reupload_classifier.py`
- [x] Tests — Pipeline/Models/Config: 6/6 grün
- [x] CI — GitHub Actions: Spec/Tests/Demos + Smoke Train/Predict (`.github/workflows/ci.yml`)

Nächste sinnvolle Schritte (kurz)
- Prediction-Export (`--export json|csv`) und reproduzierbares `--seed` + Logging.
- Hardware-Targeting: Basisgates/Coupling in Demos/CI prüfen, Transpile-Metriken festhalten.
- Realdaten-Loader (CSV/NumPy) + Evaluierung (Accuracy/AUC) und Ergebnis-Reports.