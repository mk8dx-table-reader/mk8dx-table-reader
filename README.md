# mk8dx-table-reader
Welcome to this project aming at making a full OCR program able to convert Mariokart 8 deluxe end screen results into a easy to work with text output.

## Installation

### 1. Clone this repository:
- with ssh:
```bash
git clone git@github.com:mk8dx-table-reader/mk8dx-table-reader.git
cd mk8dx-table-reader
```
- with https:
```bash
git clone https://github.com/mk8dx-table-reader/mk8dx-table-reader.git
cd mk8dx-table-reader
```

### 2. Install dependencies (choose one):

#### Option 1: GPU support (faster, requires CUDA)
Uses `easyocr` with GPU acceleration:
```bash
uv sync --extra gpu
```
#### Option B: CPU-only (lightweight, no GPU required) 
> [!Warning]
> This option does not have any purpose, as I still use ultralytics, which needs Torch. I have planned to switch every model in the code to ONNX, for onnxruntime.
Uses `torchfreeOCR` for reduced dependencies:
```bash
uv sync --extra cpu
```

### 3. Run the test to check that everything is setup well:
```bash
uv run python test_package.py
```

If you see this output, everything works:
```txt
['Edgardo_vzI', '[JOJO] KaramTNC', 'ShadowStarX', 'MINITSIKU', 'NotNiall', 'Juul-Poms', 'ShadowDeckX', 'bigtiddyGOTHgf :333', 'Yapz cars', 'targeted8dx'] ['12', '2', '7', '8', '3', '5', '4', '10', '6', '1']
```
