from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Dados(BaseModel):
    endereco: str
    bairro: str
    tipo: str
    localcometimento: str
    ano: int
    mes: int
    dia: int


@app.post("/previsao")
def previsao_total_crimes(dados: Dados):
    try:
        # Definir os mapeamentos para label encoding
        tipo_map = {
            'COLISÃO': 0,
            'CAPOTAMENTO': 1,
            'CHOQUE': 2,
            'ENGAVETAMENTO': 3,
            'TOMBAMENTO': 4,
            'ATROPELAMENTO': 5,
            'COLISÃO COM CICLISTA': 6,
            'ACID. DE PERCURSO': 7,
            'NATUREZA': 8,
            'COLISAO': 9,
            'ATROPELAMENTO ANIMAL': 10,
            '0': 11,
            'ABALROAMENTO TRANSVERSAL': 12,
            'COLISÃO TRASEIRA': 13,
            'ABALROAMENTO LONGITUDINAL': 14,
            'CHOQUE VEÍCULO PARADO': 15,
            'COLISÃO FRONTAL': 16,
            'CHOQUE OBJETO FIXO': 17,
            'COLISÃO LATERAL': 18,
            'COLISÃO TRANSVERSAL': 19,
            'QUEDA': 20,
            'ATROPELAMENTO DE PESSOA': 21,
            'Não informado': 22,
            'MONITORAMENTO': 23,
            'SEMÁFORO': 24,
            'ATROPELAMENTO DE ANIMAL': 25,
            'OUTROS APOIOS': 26,
            'APOIO EMLURB': 27,
            'FISCALIZAÇÃO': 28,
            'RENDIÇÃO': 29,
            'OUTROS': 30,
            'APOIO CELPE': 31
        }

        bairro_map = {
            'CABANGA': 0, 'SANTO AMARO': 1, 'JARDIM SÃO PAULO': 2, 'CAXANGÁ': 3,
            'BOMBA DO HEMETÉRIO': 4, 'PINA': 5, 'ÁGUA FRIA': 6, 'AFOGADOS': 7,
            'CAMPO GRANDE': 8, 'PARNAMIRIM': 9, 'SANCHO': 10, 'IMBIRIBEIRA': 11,
            'VASCO DA GAMA': 12, 'TORRE': 13, 'BOA VIAGEM': 14, 'TAMARINEIRA': 15,
            'CASA AMARELA': 16, 'CORDEIRO': 17, 'MADALENA': 18, 'SÃO JOSÉ': 19,
            'BOA VISTA': 20, 'ESTÂNCIA': 21, 'GRAÇAS': 22, 'CASA FORTE': 23,
            'FUNDÃO': 24, 'ESPINHEIRO': 25, 'IPSEP': 26, 'ROSARINHO': 27,
            'PAISSANDU': 28, 'BAIRRO DO RECIFE': 29, 'SANTANA': 30, 'AFLITOS': 31,
            'DERBY': 32, 'ILHA DO RETIRO': 33, 'ARRUDA': 34, 'SAN MARTIN': 35,
            'IBURA': 36, 'ALTO DO MANDU': 37, 'ILHA DO LEITE': 38, 'AREIAS': 39,
            'IPUTINGA': 40, 'SOLEDADE': 41, 'VÁRZEA': 42, 'COELHOS': 43,
            'TORRÕES': 44, 'BONGI': 45, 'ALTO JOSÉ BONIFÁCIO': 46, 'BEBERIBE': 47,
            'CAÇOTE': 48, 'BARRO': 49, 'PRADO': 50, 'MACAXEIRA': 51,
            'MUSTARDINHA': 52, 'COQUEIRAL': 53, 'ALTO SANTA TERESINHA': 54,
            'JAQUEIRA': 55, 'DOIS IRMÃOS': 56, 'TEJIPIÓ': 57, 'DOIS UNIDOS': 58,
            'BREJO DA GUABIRABA': 59, 'ENCRUZILHADA': 60, 'CURADO': 61,
            'SANTO ANTÔNIO': 62, 'BRASÍLIA TEIMOSA': 63, 'JOANA BEZERRA': 64,
            'NOVA DESCOBERTA': 65, 'ZUMBI': 66, 'CAJUEIRO': 67, 'JORDÃO': 68,
            'CIDADE UNIVERSITÁRIA': 69, 'ALTO JOSÉ DO PINHO': 70,
            'GUABIRABA': 71, 'APIPUCOS': 72, 'Desconhecido': 73,
            'LINHA DO TIRO': 74, 'MANGUEIRA': 75, 'ILHA JOANA BEZERRA': 76,
            'SÍTIO DOS PINTOS': 77, 'CÓRREGO DO JENIPAPO': 78, 'FABIO': 79,
            'MANGABEIRA': 80, 'TORREÃO': 81, 'ENGENHO DO MEIO': 82,
            'PORTO DA MADEIRA': 83, 'HIPÓDROMO': 84, 'TOTÓ': 85,
            'PONTO DE PARADA': 86, 'MORRO DA CONCEIÇÃO': 87, 'JIQUIÁ': 88,
            'POÇO DA PANELA': 89, 'COHAB': 90, 'PASSARINHO': 91,
            'CAMPINA DO BARRETO': 92, 'MONTEIRO': 93, 'IPESEP': 94,
            'BREJO DE BEBERIBE': 95, 'BOMBA DO HEMETERIO': 96, 'JORDAO': 97,
            'TOTO': 98, 'AGUA FRIA': 99, 'CAXANGA': 100, 'JARDIM SAO PAULO': 101,
            'SAO JOSE': 102, 'TORROES': 103, 'ALTO JOSE DO PINHO': 104,
            'TORREAO': 105, 'VARZEA': 106, 'CORREGO DO JENIPAPO': 107,
            'SANTO ANTONIO': 108, 'ESTANCIA': 109,
            'CIDADE UNIVERSITARIA': 110, 'TEJIPIO': 111, 'DOIS IRMAOS': 112,
            'GRACAS': 113, 'BRASILIA TEIMOSA': 114,
            'MORRO DA CONCEICAO': 115, 'SITIO DOS PINTOS': 116,
            'POCO DA PANELA': 117, 'CACOTE': 118, 'ALTO JOSE BONIFACIO': 119,
            '0': 120, 'JIQUIA': 121
        }

        endereco_map = {
            'AV SUL': 0, 'RUA BARROS BARRETO': 1, 'AV PIRACICABA': 2,
            'RUA NETO CAMPELO': 3, 'RUA SUBIDA DO PLATO': 4, 'RUA JOAO CARNEIRO DA CUNHA': 5,
        }

        localcometimento_map = {
            'Desconhecido': 0, 'AV. ENG. ABDIAS DE CARVALHO, SEMAFORO 328.': 1,
            'AV. GOV. AGAMENON MAGALHAES, SEMAFORO 173.': 2, 'RUA DO JARDIM, AO LADO AO N. 22': 3,
            'RUA JOAQUIM NABUCO, APOS AO SEMAFORO N. 174 SENTIDO SUBURBIO': 4,
            'AVENIDA SUL, SOB AO SEMAFORO N. 260': 5,
        }

        # Aplicar o mapeamento aos valores de entrada
        tipo_codificado = tipo_map.get(dados.tipo, -1)
        bairro_codificado = bairro_map.get(dados.bairro, -1)
        endereco_codificado = endereco_map.get(dados.endereco, -1)
        localcometimento_codificado = localcometimento_map.get(dados.localcometimento, -1)

        # Verificar se algum valor não foi encontrado nos mapeamentos
        if -1 in [tipo_codificado, bairro_codificado, endereco_codificado, localcometimento_codificado]:
            return {"error": "Um ou mais valores de entrada são inválidos ou não mapeados."}

        # Criar DataFrame para entrada no modelo
        input_data = pd.DataFrame({
            'endereco': [endereco_codificado],
            'bairro': [bairro_codificado],
            'tipo': [tipo_codificado],
            'localcometimento': [localcometimento_codificado],
            'ano': [dados.ano],
            'mes': [dados.mes],
            'dia': [dados.dia],
        })

        # Carregar o modelo RandomForest
        randomforest = joblib.load('src/resources/randomforest.pkl')
        previsao = randomforest.predict(input_data)

        # Interpretar a previsão
        if previsao == 0:
            result = "Com vítima"
        elif previsao == 1:
            result = "Sem vítima"
        else:
            result = "Previsão desconhecida"

        return {"Previsão": result}

    except Exception as e:
        return {"error": str(e)}
