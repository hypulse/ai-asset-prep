# image-cut-fit

AI 생성 이미지나 에셋을 활용해 빠르게 게임이나 앱 데모를 만들고 싶어서 개발한 로컬 앱입니다.
배경과 빈 여백을 정리해 PNG 스프라이트로 저장하고, 한 장에 모인 스프라이트도 개별 이미지로 분리할 수 있습니다.

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

macOS에서는 `Image Cut Fit.command`를 더블클릭해 실행할 수 있습니다.

`rembg`는 첫 실행 시 모델 파일을 다운로드할 수 있습니다.

## CLI

Streamlit 없이 같은 이미지 처리 기능을 터미널에서 실행할 수 있습니다.

```bash
python cli.py crop input.png --model none --output outputs/cropped.png
python cli.py fit input.png --width 1280 --height 720 --resize-mode contain_center --output outputs/fitted.png
python cli.py sheet make a.png b.png --gap 0 --scale 1 --resampling nearest --output outputs/sheet.png --metadata outputs/sheet.json
python cli.py sheet recover sheet.png --scale 0.25 --alpha 16 --min-area 16 --output-dir outputs/sprites
python cli.py tiles guide --tile-size 64 --prefix ground --output outputs/ground_tileset_guide_64px.png
python cli.py tiles slice guide.png --tile-size 64 --prefix ground --output-dir outputs/tiles
```

CLI는 처리 결과를 JSON으로 출력합니다. 출력 파일명이 이미 있으면 `_2`, `_3`처럼 suffix를 붙여 덮어쓰지 않습니다.

기본값은 앱과 맞춰져 있습니다.

- 최대 출력 크기: `8192px`
- 기본 alpha threshold: `16`
- 기본 스케일 방식: `nearest`
- 기본 배경 제거 모델: `u2net`
- 모델 없이 원본 alpha만 기준으로 자르기: `--model none`

캔버스 기반 클립보드 붙여넣기, 브러시, 그림 지우개, 이미지 지우개는 Streamlit UI 전용입니다. CLI/MCP에서는 파일 경로 기반의 핵심 처리 기능만 제공합니다.

## MCP

Codex가 이 프로젝트의 이미지 처리 기능을 직접 호출할 수 있도록 stdio MCP 서버를 제공합니다.

```bash
python mcp_server.py
```

Codex MCP 설정 예시:

```json
{
  "mcpServers": {
    "image-cut-fit": {
      "command": "/Users/seungjae/Codes/image-cut-fit/.venv/bin/python",
      "args": ["/Users/seungjae/Codes/image-cut-fit/mcp_server.py"],
      "cwd": "/Users/seungjae/Codes/image-cut-fit"
    }
  }
}
```

제공 도구:

- `crop_image`
- `fit_image`
- `make_sprite_sheet`
- `recover_sprite_sheet`
- `make_tileset_guide`
- `slice_tileset`

## 탭별 기능

- `자동 배경 제거/리사이즈`: 이미지 배경 제거, 투명 영역 crop, 수동 지우기, 회전, padding, 배경색, 최종 크기 맞춤을 처리합니다.
- `수동 자르기/그리기`: crop 박스를 직접 조정하고, 확대 보기에서 브러시, 그림 지우개, 이미지 지우개를 사용한 뒤 지정한 가로/세로 크기로 내보냅니다.
- `스프라이트 시트 생성/복구`: 여러 이미지를 한 장의 스프라이트 시트로 합치거나, 기존 투명 배경 시트에서 개별 스프라이트를 다시 분리합니다.
- `타일셋 가이드/타일 추출`: 3x3 타일셋 가이드 PNG를 만들고, 완성된 가이드 이미지에서 타일 PNG를 잘라 ZIP으로 내보냅니다.

각 탭에서 이미지를 업로드하거나 클립보드에서 붙여넣은 뒤 결과를 PNG로 다운로드하거나 `outputs/`에 저장할 수 있습니다.
