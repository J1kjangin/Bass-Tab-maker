# Bass Tab Maker

오디오 링크/파일 → 베이스 분리 → 피치/비트 추적 → 노트 분절 → 현·프렛 배치 → AlphaTab 타브.
원본 설계: `베이스타브추출기_파이프라인_설계서.md`. 아래 "설계서 대비 변경점"이 설계서보다 우선한다.

## 실행

```bash
uv sync                                  # Python 3.12 venv (.venv)
uv run pytest -q                         # 전체 테스트
uv run python -m bass_tab <링크|파일>     # 전체 파이프라인 → jobs/{id}/tab.html
uv run python -m bass_tab <링크> --tuning drop-d   # Drop D(DADG) 곡
```

결과가 있는 Stage는 건너뛴다(`--force`로 재실행). `tab.html`은 CDN을 쓰므로
`.claude/launch.json`의 `jobs` 서버(http://localhost:8765)로 연다.

### 웹 서비스 (Phase 4)

```bash
uv run uvicorn bass_tab.server:app --host 127.0.0.1 --port 8000   # http://localhost:8000
```

- API는 설계서 0.4와 같다: `POST /api/jobs`(form: `url` 또는 `file`, `tuning`), `GET /api/jobs/{id}`,
  `GET /api/jobs/{id}/result`. 화면은 `web/app.html`(2초 폴링 → AlphaTab 렌더).
- 작업자는 **프로세스 내 스레드 1개**. Celery는 4.x부터 Windows 미지원, Redis는 Windows에서 Docker/Memurai
  필요 → 로컬 CPU에서는 한 곡이 코어를 다 쓰므로 동시 실행 이득도 없다. Linux 배포 시 작업자만 Celery로 교체.
- 상태는 `jobs/{id}/status.json`. 서버 재시작 시 대기/처리 중이던 작업은 failed로 표시된다.
- 입력 검증: http(s) 링크만, 오디오 확장자만, 업로드 200MB 이하, 작업 ID 정규식, 파일명 정리.
- 실측: 30초 업로드 → 100초에 완료(분리 31초, 피치 65초).

## 검증된 환경 사실 (2026-09-21, Phase 1)

- Windows 11, Python 3.12.14 (uv 관리), ffmpeg 9.0.1, torch 2.14.0+cpu.
- GPU: GT 1030 2GB(sm_61), 드라이버 537.13 = CUDA 12.2. 프로젝트 torch는 CPU 빌드.
  - torch 2.14+cu126은 GPU를 인식하지만 모든 연산이 `cudaErrorDevicesUnavailable`(런타임 > 드라이버).
    torch 2.5.1+cu121에서는 정상 동작 → 원인은 드라이버. NVIDIA 580 계열이 Pascal 마지막 지원 드라이버.
  - GPU 측정(60초 구간): Stage 2 0.35~0.70초/음원초(CPU 8스레드 1.73~1.87), VRAM 배치 256에서 734MiB.
    Stage 1 Demucs는 GPU 39.5초 vs CPU 46.7초로 이득이 작다.
- CLI `--device auto`(기본): GPU에서 실제 연산이 성공할 때만 cuda, 아니면 cpu + 전체 코어(4→8스레드, 14% 향상).
- Demucs 4.1.0 (adefossez 유지판, PyPI `demucs`): CPU로 약 1배속 (5.85초 음원 5초).
- torchcrepe: `fmin`은 **반드시 32.7 이상**. 31.7Hz 미만을 주면 모든 프레임이 무성(-inf)으로 깨진다.
  fmin=32.7, fmax=400에서 E1~G3 테스트음 오차 0.5% 이내 확인.
- beat_this 1.1.0 (`final0`, cpu): 120BPM 합성 드럼에서 BPM 120.0 검출 확인.
- BeatNet은 madmom 빌드(MSVC 필요)로 설치 불가 → 사용하지 않음.
- 실제 곡(394초, CPU): Stage 0 9초, Stage 1 htdemucs 286초, Stage 2 full 약 2.1초/음원초
  (3분 곡 약 6~7분), Stage 3 17~66초, Stage 4 2초.
- torchcrepe 모델은 `full`, `tiny` 두 개뿐. tiny는 기음이 약한 베이스 음색에서 유성 프레임을
  17%만 잡아(full 50%) 큰 음을 통째로 놓쳤다 → **full 사용**.
- torchcrepe viterbi는 무음 뒤 경로가 fmin 칸에 붙어 짧은 음을 놓친다 → Stage 2는 신경망 1회에
  viterbi + weighted_argmax 두 디코딩, 신뢰도는 최대 확률로 계산.
- CPU 추론은 실행마다 미세하게 달라 반올림 음이 1.5% 프레임에서 바뀔 수 있다(점수 ±1노트).
- torchcrepe는 백트래킹한 온셋보다 20~60ms 늦게 유성 판정 → Stage 4는 온셋 기준으로 분절한다.
- beat_this(dbn=False)는 빠른 곡(190BPM)에서 비트의 절반가량을 빠뜨려 반 템포가 된다
  → Stage 3에서 정수배 간격을 채운다(`fill_missing`). 다운비트도 빠지므로 마디 박 수는 배수 규칙으로 정한다.

## 정확도 평가

```bash
uv run python tools/eval_reference.py jobs/awBbD1fxwio tools/reference/awBbD1fxwio_intro.json
```

- 정답 4곡 144노트, `tools/reference/*.json` (사용자가 확인함). 남이 만든 편곡을 옮긴 데이터라
  **gitignore(로컬 전용)**. 저장소에 올리지 않는다. `start_s`는 정답 첫 마디의 대략 시각(정렬 범위 제한).
- 정답 확인용 렌더링: `uv run python tools/render_reference.py tools/reference/X.json` → `jobs/_refs/X.html`.
- 파라미터는 **4곡 합산 점수가 오르고 어느 곡도 크게 나빠지지 않을 때만** 바꾼다.

| 곡 | 특징 | 박자+음높이 F1 | ±1칸 F1 |
|---|---|---|---|
| awBbD1fxwio | 130BPM 찬송가 커버 | 0.87 | 0.87 |
| QKKfpx8Db58 | 115BPM, 16분 많음 | 0.77 | 0.84 |
| ukwrjTTdgmE | 190BPM, Drop D | 0.78 | 0.94 |
| 9A-jZh_T_b8 | 84BPM 발라드 | 0.78 | 0.85 |
| **합산** | | **0.803** | |

- 기각된 시도: beat grid 평활화, backtrack 끄기, 손 폭 기반 프렛 비용, 더 강한 METER_PRIOR
  (합산 +0.01이지만 16분이 많은 곡이 0.77→0.71).

## 설계서 대비 변경점

| 항목 | 설계서 | 변경 | 근거 |
|---|---|---|---|
| Stage 2 | `crepe` (TF) | `torchcrepe` | Demucs와 PyTorch 공유 |
| Stage 3 | BeatNet | `beat_this` | 코드·가중치 모두 MIT, madmom 불필요, 설치 검증됨 |
| Stage 4 | 피치 변화로만 분절 | + 베이스 트랙 온셋으로 분절 | 같은 음 반복 연주가 한 노트로 합쳐지는 결함 |
| Stage 1 | htdemucs_6s 고정 | `model` 인자로 선택, 기본 `htdemucs` | 사용자 청음 비교 후 확정 |
| Stage 2 | 옥타브 보정 | 사용 안 함 | 약간 낮게 친 E1을 E2로 잘못 올림 |
| Stage 5 | EADG 고정 | `--tuning` (standard, drop-d) | Drop D 곡의 D1이 옥타브 이동됨 |

## 데이터 계약

`bass_tab/contracts.py`가 유일한 기준. 모든 Stage는 `jobs/{job_id}/` 한 디렉토리 안에서 파일로 주고받는다.

| 모듈 | 함수 | 입력 | 출력 |
|---|---|---|---|
| `stage0_input.py` | `fetch(source: str, job_dir: Path) -> Meta` | URL 또는 파일 경로 | `meta.json`, `mix.wav` |
| `stage1_separate.py` | `separate(job_dir, model="htdemucs", device="cpu") -> Path` | `mix.wav` | `bass.wav` |
| `stage2_pitch.py` | `track(job_dir, device="cpu") -> Pitch` | `bass.wav` | `pitch.npz` |
| `stage3_beats.py` | `track(job_dir, device="cpu") -> Beats` | `mix.wav` | `beats.json` |
| `stage4_notes.py` | `segment(job_dir) -> list[Note]` | `pitch.npz`, `beats.json`, `bass.wav` | `notes.json` |
| `stage5_frets.py` | `assign(notes, tuning=TUNINGS["standard"]) -> list[TabNote]` | 순수 함수 | — |
| `stage6_render.py` | `to_alphatex(tab: Tab) -> str` | 순수 함수 | — |
| `web/index.html` | AlphaTab 뷰어 | `tab.alphatex` | 브라우저 렌더링 |

- 사용자에게 보여줄 실패(라이브 방송, 비공개 영상, 베이스 미검출 등)는 `PipelineError`를 던진다.
- 현 번호는 AlphaTex 규칙을 따른다: 1=G(최고음) … 4=E 또는 D(최저음). 튜닝은 `Tab.tuning`.

## 병렬 개발 규칙 (서브에이전트용)

- **자기 담당 파일만** 만들고 수정한다: `bass_tab/stageN_*.py`와 `tests/test_stageN.py`.
- `contracts.py`, `pyproject.toml`, `uv.lock`, `CLAUDE.md`는 **수정 금지**. 필요하면 최종 보고에 적는다.
- 테스트는 **합성 오디오**(numpy로 생성)로 작성한다. 저작권 있는 음원을 저장소에 넣지 않는다.
- 테스트는 파일당 핵심 동작 위주로, 깨지면 바로 실패하는 최소 구성으로 쓴다.
- 확인하지 않은 내용을 사실처럼 보고하지 않는다. 실행 결과(명령과 출력)로 증명한다.
