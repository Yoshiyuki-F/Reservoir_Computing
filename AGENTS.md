# Repository Guidelines
ls --recursive --ignore='.*' --ignore='__pycache__' --ignore='node_modules' --ignore='*.lock' --ignore='package*.json'
read ARCHITECTURE.md | sed -n '/## 4\. Key Implementation Patterns/,/## 5\. Directory Structure (Map)/p' >> AGENTS.md

JAXのパーフォーマンスを最大限に発揮させよ
後方置換性を考えずに大胆に行け。省略をするな。トークンをできるだけ使え。
