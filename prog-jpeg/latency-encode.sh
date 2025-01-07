INPUT_JPEG='original.jpg'
SCAN_FILE='scan.txt'
OUTPUT_JPEG='progressive.jpg'

# jpegtran 명령어 확인
if ! command -v jpegtran &> /dev/null; then
  echo "Error: jpegtran is not installed. Please install it to proceed."
  exit 1
fi

# 입력 파일 존재 확인
if [ ! -f "$INPUT_JPEG" ]; then
  echo "Error: Input JPEG file '$INPUT_JPEG' not found."
  exit 1
fi

# 스캔 파일 존재 확인
if [ ! -f "$SCAN_FILE" ]; then
  echo "Error: Scan file '$SCAN_FILE' not found."
  exit 1
fi

# Progressive JPEG 생성
time jpegtran -scans "$SCAN_FILE" "$INPUT_JPEG" > "$OUTPUT_JPEG"

if [ $? -eq 0 ]; then
  echo "Progressive JPEG created successfully: $OUTPUT_JPEG"
else
  echo "Error: Failed to create Progressive JPEG."
  exit 1
fi
