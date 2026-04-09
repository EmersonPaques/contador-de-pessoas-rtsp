# contador-de-pessoas-rtsp

Contagem de entrada e saida de pessoas com streams RTSP, ROI por camera e visualizacao em mosaico.

## O que este projeto faz

- exibe ate 16 cameras RTSP em mosaico
- conta `IN` e `OUT` por camera
- permite configurar ROI poligonal por canal
- carrega DVRs, canais, usuarios e senhas a partir de `.env`
- evita credenciais no codigo-fonte

## Principais recursos

- mosaico multitelas para monitoramento simultaneo
- reconexao automatica em caso de queda do stream
- configuracao individual por camera
- suporte a modelos YOLO `.pt`
- arquivo `.env.example` pronto para servir de modelo

## Estrutura de configuracao

O projeto usa um arquivo `.env` local para armazenar:

- credenciais RTSP
- configuracoes gerais de execucao
- cadastro dos slots `CAM_01` ate `CAM_16`
- ROI e linhas de contagem por camera

O arquivo `.env` nao deve ser versionado. O repositorio inclui apenas o `.env.example`.

## Instalacao

```bash
pip install -r requirements_counter.txt
```

No PowerShell:

```powershell
Copy-Item .env.example .env
```

Depois, edite o `.env` com os dados do seu ambiente.

## Exemplo de cadastro no `.env`

```dotenv
RTSP_USERNAME=seu_usuario
RTSP_PASSWORD=sua_senha
RTSP_PORT=554
RTSP_SUBTYPE=0
RTSP_PATH_TEMPLATE=rtsp://{username}:{password}@{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}

CAM_01_ENABLED=true
CAM_01_NAME=Entrada Loja
CAM_01_HOST=192.168.1.10
CAM_01_CHANNEL=1

CAM_02_ENABLED=true
CAM_02_NAME=Saida Loja
CAM_02_HOST=192.168.1.11
CAM_02_CHANNEL=2
```

## Execucao

No PowerShell:

```powershell
.\.venv312\Scripts\python.exe .\main.py
```

Ou:

```powershell
python .\main.py
```

## Documentacao

Guia detalhado:

- [README_counter.md](README_counter.md)

Esse arquivo detalha:

- como montar o `.env`
- como cadastrar DVRs e canais
- como configurar ROI e linhas
- dicas de desempenho e troubleshooting

## Seguranca

- nao publique o arquivo `.env`
- nao deixe IPs reais, usuarios ou senhas em arquivos versionados
- revise sempre `README`, `.env.example` e codigo antes de subir para outro repositorio

## Licenciamento e dependencias

- este repositório contem o codigo de integracao da aplicacao
- dependencias de terceiros, como `ultralytics`, permanecem sob suas proprias licencas
- este projeto nao inclui licenca propria para redistribuir automaticamente componentes de terceiros
- revise [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) antes de reutilizar ou redistribuir
