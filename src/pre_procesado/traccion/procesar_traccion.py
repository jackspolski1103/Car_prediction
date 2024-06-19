import re

def buscar_palabra(texto, palabra):
    patron = r'\b' + palabra + r'\b'
    matches = re.findall(patron, texto)
    return len(matches) > 0

 

def preprocesar_tracciones(df):
    lista_tracciones = [
                '4X4',
                '4WD',
                'SAWD',
                '4MATIC',
                '4MOTION',
                '4IMOTION',
                'XDRIVE',
                '4 X 4',
                'ALL GRIP',
                'ALL4'
    ]
    versiones = df['Versión'].str.upper()
    #cambiar todos los nan por 'OTRO'
    versiones = versiones.fillna('OTRO')
    tracciones = []
    for version in versiones:
        traccion = 'OTRO'
        for t in lista_tracciones:
            
            if buscar_palabra(version, t):
                traccion = '4X4'
        tracciones.append(traccion)

    df['Tracción'] = tracciones
    return df
