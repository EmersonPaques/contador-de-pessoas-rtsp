# People Counter RTSP Mosaic

Sistema de contagem de pessoas em tempo real com Ultralytics YOLO e OpenCV, preparado para exibir ate 16 cameras em mosaico e carregar configuracoes a partir de um arquivo `.env`.

## Visao geral

- Mosaico com ate 16 cameras RTSP
- Contagem separada de `IN` e `OUT` por camera
- ROI poligonal por camera
- Reconexao automatica quando um stream cai
- Configuracao por `.env`, sem credenciais no codigo
- `.env` ignorado no git para evitar vazamento de dados sensiveis

## Requisitos

- Python 3.10+
- GPU recomendada para melhor desempenho
- Streams RTSP acessiveis na rede local

## Instalacao

1. Instale as dependencias:

```bash
pip install -r requirements_counter.txt
```

2. Copie o arquivo de exemplo para `.env`:

No PowerShell:

```powershell
Copy-Item .env.example .env
```

3. Edite o arquivo `.env` com os dados do seu ambiente.

## Como funciona a configuracao por `.env`

O projeto le todas as configuracoes em tempo de execucao a partir do arquivo `.env`.

Credenciais globais:

- `RTSP_USERNAME`
- `RTSP_PASSWORD`
- `RTSP_PORT`
- `RTSP_SUBTYPE`
- `RTSP_PATH_TEMPLATE`

Configuracoes gerais:

- `MODEL_PATH`
- `CONFIDENCE`
- `IMG_SIZE`
- `SHOW_WINDOW`
- `WINDOW_NAME`
- `TRACK_TIMEOUT`
- `LINE_OFFSET`
- `MOSAIC_TILE_WIDTH`
- `MOSAIC_TILE_HEIGHT`

Padroes para novas cameras:

- `CAM_DEFAULT_REFERENCE_SIZE`
- `CAM_DEFAULT_LINE_IN_Y`
- `CAM_DEFAULT_LINE_OUT_Y`
- `CAM_DEFAULT_ROI_POLYGON`

Slots de camera:

- `CAM_01_*` ate `CAM_16_*`
- cada slot pode ser ativado ou desativado com `CAM_XX_ENABLED=true|false`

## Exemplo de `.env`

Use apenas valores ficticios no repositorio. Preencha os valores reais somente no seu `.env` local.

```dotenv
MODEL_PATH=yolov12n.pt
CONFIDENCE=0.35
IMG_SIZE=640
SHOW_WINDOW=true
WINDOW_NAME=People Counter RTSP Mosaic
RECONNECT_DELAY_SECONDS=5
PERSON_CLASS_ID=0
TRACK_TIMEOUT=30
LINE_OFFSET=20
APP_MAX_CAMERAS=16
MOSAIC_TILE_WIDTH=640
MOSAIC_TILE_HEIGHT=360

RTSP_USERNAME=seu_usuario
RTSP_PASSWORD=sua_senha
RTSP_PORT=554
RTSP_SUBTYPE=0
RTSP_PATH_TEMPLATE=rtsp://{username}:{password}@{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}

CAM_DEFAULT_REFERENCE_SIZE=1920,1080
CAM_DEFAULT_LINE_IN_Y=750
CAM_DEFAULT_LINE_OUT_Y=400
CAM_DEFAULT_ROI_POLYGON=700,120;1750,120;1910,1020;850,1020

CAM_01_ENABLED=true
CAM_01_NAME=Entrada Loja
CAM_01_HOST=192.168.1.10
CAM_01_CHANNEL=1

CAM_02_ENABLED=true
CAM_02_NAME=Saida Loja
CAM_02_HOST=192.168.1.11
CAM_02_CHANNEL=2
```

## Como cadastrar DVRs e cameras

Cada camera ocupa um slot de `CAM_01` a `CAM_16`.

Campos principais por slot:

- `CAM_XX_ENABLED`: ativa ou desativa a camera
- `CAM_XX_NAME`: nome exibido no mosaico
- `CAM_XX_HOST`: IP ou hostname do DVR/NVR
- `CAM_XX_CHANNEL`: canal da camera no DVR

Campos opcionais por slot:

- `CAM_XX_SUBTYPE`
- `CAM_XX_REFERENCE_SIZE`
- `CAM_XX_LINE_IN_Y`
- `CAM_XX_LINE_OUT_Y`
- `CAM_XX_ROI_POLYGON`

Exemplo para adicionar uma terceira camera:

```dotenv
CAM_03_ENABLED=true
CAM_03_NAME=Caixa 01
CAM_03_HOST=192.168.1.12
CAM_03_CHANNEL=3
```

Se voce nao informar ROI ou linhas por camera, o sistema usa os valores padrao `CAM_DEFAULT_*`.

## ROI e linhas de contagem

Cada camera pode ter:

- uma ROI poligonal em `CAM_XX_ROI_POLYGON`
- uma linha de entrada em `CAM_XX_LINE_IN_Y`
- uma linha de saida em `CAM_XX_LINE_OUT_Y`

Formato da ROI:

```dotenv
CAM_01_ROI_POLYGON=700,120;1750,120;1910,1020;850,1020
```

Cada ponto e definido como `x,y` e os pontos sao separados por `;`.

## Execucao

No PowerShell:

```powershell
.\.venv312\Scripts\python.exe .\main.py
```

Ou com Python global:

```powershell
python .\main.py
```

## Desempenho

Para melhorar FPS:

- reduza `IMG_SIZE` para `640` ou `512`
- use `RTSP_SUBTYPE=1` se o DVR tiver substream
- use menos cameras simultaneas
- prefira ROI ajustada para reduzir deteccoes desnecessarias

## Seguranca e publicacao

- nunca publique o arquivo `.env`
- nunca deixe IPs reais, usuarios ou senhas em `README`, `main.py` ou `.env.example`
- o `.env` ja esta listado no `.gitignore`
- antes de subir para outro repositorio, revise se nao ha credenciais em arquivos versionados

## Troubleshooting

- Stream nao conecta:
  verifique IP, canal, usuario, senha, porta e formato da URL RTSP
- FPS baixo:
  reduza `IMG_SIZE` e teste `RTSP_SUBTYPE=1`
- Contagem incorreta:
  ajuste `CAM_XX_LINE_IN_Y`, `CAM_XX_LINE_OUT_Y` e `CAM_XX_ROI_POLYGON`
- Camera fica preta no mosaico:
  confirme se o stream abre no VLC e se o canal esta correto

## Licenciamento e dependencias

- este repositório contem o codigo de integracao da aplicacao
- bibliotecas e modelos de terceiros seguem licencas proprias
- consulte [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) antes de redistribuir o projeto ou empacotar dependencias
