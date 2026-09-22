# Bass Tab Maker

오디오 링크/파일 → 베이스 분리 → 피치/비트 추적 → 노트 분절 → 현·프렛 배치 → AlphaTab 타브.
원본 설계: `베이스타브추출기_파이프라인_설계서.md`. 아래 "설계서 대비 변경점"이 설계서보다 우선한다.

## 실행

```bash
uv sync                                  # Python 3.12 venv (.venv)
uv run pytest -q                         # 전체 테스트
uv run python -m bass_tab <링크|파일>     # 전체 파이프라인 → jobs/{id}/tab.html
```

결과가 있는 Stage는 건너뛴다(`--force`로 재실행). `tab.html`은 CDN을 쓰므로
`.claude/launch.json`의 `jobs` 서버(http://localhost:8765)로 연다.

## 검증된 환경 사실 (2026-09-21, Phase 1)

- Windows 11, Python 3.12.14 (uv 관리), ffmpeg 9.0.1, torch 2.14.0+cpu.
- GPU: GT 1030 2GB. 현재 torch는 CPU 빌드 → **모든 Stage는 `device="cpu"` 기본값**으로 작성.
- Demucs 4.1.0 (adefossez 유지판, PyPI `demucs`): CPU로 약 1배속 (5.85초 음원 5초).
- torchcrepe: `fmin`은 **반드시 32.7 이상**. 31.7Hz 미만을 주면 모든 프레임이 무성(-inf)으로 깨진다.
  fmin=32.7, fmax=400에서 E1~G3 테스트음 오차 0.5% 이내 확인.
- beat_this 1.1.0 (`final0`, cpu): 120BPM 합성 드럼에서 BPM 120.0 검출 확인.
- BeatNet은 madmom 빌드(MSVC 필요)로 설치 불가 → 사용하지 않음.
- 실제 곡(394초, CPU): Stage 0 9초, Stage 1 htdemucs 286초, Stage 2 tiny 37초
  (full은 2.05초/음원초라 약 13분), Stage 3 17~66초, Stage 4 2초.
- torchcrepe는 백트래킹한 온셋보다 20~60ms 늦게 유성 판정 → Stage 4는 온셋 기준으로 분절한다.

## 정확도 평가

```bash
uv run python tools/eval_reference.py jobs/awBbD1fxwio tools/reference/awBbD1fxwio_intro.json
```

- 정답: 사용자가 준 타브 이미지의 인트로 7마디(41노트), `tools/reference/`.
  남이 만든 편곡을 옮긴 데이터라 **gitignore(로컬 전용)**. 저장소에 올리지 않는다.
- Stage 4/5 파라미터는 **이 점수를 올릴 때만** 바꾼다. 현재: 박자+음높이 일치 F1 0.90,
  음높이 39/41, 줄·프렛 일치 29/37.
- 기각된 시도(점수 개선 없음): beat grid 평활화, backtrack 끄기, 손 폭(span) 기반 프렛 비용.
- 정답이 한 곡 41노트뿐이라 과적합 위험이 있다. 정답 곡이 늘면 reference를 추가한다.

## 설계서 대비 변경점

| 항목 | 설계서 | 변경 | 근거 |
|---|---|---|---|
| Stage 2 | `crepe` (TF) | `torchcrepe` | Demucs와 PyTorch 공유 |
| Stage 3 | BeatNet | `beat_this` | 코드·가중치 모두 MIT, madmom 불필요, 설치 검증됨 |
| Stage 4 | 피치 변화로만 분절 | + 베이스 트랙 온셋으로 분절 | 같은 음 반복 연주가 한 노트로 합쳐지는 결함 |
| Stage 1 | htdemucs_6s 고정 | `model` 인자로 선택, 기본 `htdemucs` | 사용자 청음 비교 후 확정 |

## 데이터 계약

`bass_tab/contracts.py`가 유일한 기준. 모든 Stage는 `jobs/{job_id}/` 한 디렉토리 안에서 파일로 주고받는다.

| 모듈 | 함수 | 입력 | 출력 |
|---|---|---|---|
| `stage0_input.py` | `fetch(source: str, job_dir: Path) -> Meta` | URL 또는 파일 경로 | `meta.json`, `mix.wav` |
| `stage1_separate.py` | `separate(job_dir, model="htdemucs", device="cpu") -> Path` | `mix.wav` | `bass.wav` |
| `stage2_pitch.py` | `track(job_dir, device="cpu") -> Pitch` | `bass.wav` | `pitch.npz` |
| `stage3_beats.py` | `track(job_dir, device="cpu") -> Beats` | `mix.wav` | `beats.json` |
| `stage4_notes.py` | `segment(job_dir) -> list[Note]` | `pitch.npz`, `beats.json`, `bass.wav` | `notes.json` |
| `stage5_frets.py` | `assign(notes: list[Note]) -> list[TabNote]` | 순수 함수 | — |
| `stage6_render.py` | `to_alphatex(tab: Tab) -> str` | 순수 함수 | — |
| `web/index.html` | AlphaTab 뷰어 | `tab.alphatex` | 브라우저 렌더링 |

- 사용자에게 보여줄 실패(라이브 방송, 비공개 영상, 베이스 미검출 등)는 `PipelineError`를 던진다.
- 현 번호는 AlphaTex 규칙을 따른다: 1=G, 2=D, 3=A, 4=E.

## 병렬 개발 규칙 (서브에이전트용)

- **자기 담당 파일만** 만들고 수정한다: `bass_tab/stageN_*.py`와 `tests/test_stageN.py`.
- `contracts.py`, `pyproject.toml`, `uv.lock`, `CLAUDE.md`는 **수정 금지**. 필요하면 최종 보고에 적는다.
- 테스트는 **합성 오디오**(numpy로 생성)로 작성한다. 저작권 있는 음원을 저장소에 넣지 않는다.
- 테스트는 파일당 핵심 동작 위주로, 깨지면 바로 실패하는 최소 구성으로 쓴다.
- 확인하지 않은 내용을 사실처럼 보고하지 않는다. 실행 결과(명령과 출력)로 증명한다.
