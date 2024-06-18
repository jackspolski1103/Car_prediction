import numpy as np
import pandas as pd 
import Levenshtein as lev 


def preprocesar_vendedor(df): 
    tipos = [
        "1987",
        "1992",
        "1993",
        "1994",
        "1995",
        "1996",
        "1997",
        "1998",
        "1999",
        "2000",
        "2001",
        "2002",
        "2003",
        "2004",
        "2005",
        "2006",
        "2007",
        "2008",
        "2009",
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
    ]
    threshold = 3 # cantidad de letras distintas que puede tener como máximo 
    vendedor = df['Tipo de vendedor'].str.split().str[0].str.upper()
    n = len(vendedor)  
    for i in range(n):
        min_dist = 100
        aux = "" 
        for c in tipos: 
            if pd.isnull(vendedor[i]) or vendedor[i] is pd.NA:  
                vendedor[i] = "NO ESPESIFICA"  
            dist = lev.distance(vendedor[i], c) 
            if dist < min_dist:
                aux = c  
                min_dist = dist 
        vendedor[i] = aux 
        if min_dist > threshold:
            vendedor[i] = "OTRO"
    df['Tipo de vendedor'] = vendedor
    return 

