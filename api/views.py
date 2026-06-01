from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.apps import apps
from django.core.cache import cache
import json
import time
import random
import re
import numpy as np
from prometheus_client import Histogram, Counter

# Custom MLOps Prometheus metrics
LATENCY_HISTOGRAM = Histogram('ml_itinerary_generation_latency_seconds', 'Latency of itinerary generation and ML inference in seconds')
SENTIMENT_COUNTER = Counter('ml_sentiment_classifications_total', 'Total number of sentiment classification requests', ['sentiment_label'])
ITINERARY_COUNTER = Counter('ml_itineraries_generated_total', 'Total number of itineraries generated')

# Dictionary mapping common typos/informal Indonesian words to formal/expected words
INDONESIAN_SPELL_CORRECTION = {
    # ─── Konjungsi Waktu (Temporal Conjunctions) ───
    # ketika
    'ktika': 'ketika', 'ktk': 'ketika', 'ketka': 'ketika', 'ketikaj': 'ketika', 'ketikay': 'ketika',
    'ktikaa': 'ketika', 'ketikka': 'ketika', 'ketiak': 'ketika', 'ktiak': 'ketika', 'ketikak': 'ketika',
    'ketiakj': 'ketika', 'ketiika': 'ketika', 'ketiikka': 'ketika', 'kketika': 'ketika', 'ketkika': 'ketika',
    'ktikk': 'ketika', 'ktikaj': 'ketika', 'ktikay': 'ketika', 'ketyka': 'ketika', 'ketykaj': 'ketika', 'ketykay': 'ketika',
    'when': 'ketika', 'whn': 'ketika', 'wenn': 'ketika', 'whenn': 'ketika', 'whenj': 'ketika', 'wheny': 'ketika',
    'while': 'ketika', 'whl': 'ketika', 'whilee': 'ketika', 'whilej': 'ketika', 'whiley': 'ketika',
    # saat
    'sat': 'saat', 'saatt': 'saat', 'saaat': 'saat', 'satj': 'saat', 'saty': 'saat', 'saattj': 'saat',
    'saatty': 'saat', 'saast': 'saat', 'saata': 'saat', 'ssat': 'saat', 'saattt': 'saat', 'saattjj': 'saat',
    'saattyy': 'saat', 'satt': 'saat', 'saatks': 'saat', 'saatjks': 'saat',
    # pas
    'pass': 'pas', 'ps': 'pas', 'pasj': 'pas', 'pasy': 'pas', 'pasjks': 'pas', 'passs': 'pas',
    'pasas': 'pas', 'passy': 'pas', 'passj': 'pas',
    # sebelum
    'sblm': 'sebelum', 'sebelm': 'sebelum', 'seblm': 'sebelum', 'sbelm': 'sebelum', 'sebelumm': 'sebelum',
    'sblmm': 'sebelum', 'sebelunj': 'sebelum', 'sebeluny': 'sebelum', 'sbelum': 'sebelum', 'sbelumj': 'sebelum',
    'sbelumy': 'sebelum', 'sebellom': 'sebelum', 'seblom': 'sebelum', 'sblom': 'sebelum', 'sblmj': 'sebelum',
    'sblmy': 'sebelum', 'sebellm': 'sebelum', 'sebellumm': 'sebelum', 'sebluum': 'sebelum', 'sbelumm': 'sebelum',
    'sbelunj': 'sebelum', 'sbeluny': 'sebelum', 'sbln': 'sebelum', 'sebeln': 'sebelum', 'sebelun': 'sebelum',
    'sebelunk': 'sebelum', 'sbelunk': 'sebelum',
    'before': 'sebelum', 'b4': 'sebelum', 'bfore': 'sebelum', 'befor': 'sebelum', 'beforee': 'sebelum',
    'beforej': 'sebelum', 'beforey': 'sebelum',
    # setelah
    'stlh': 'setelah', 'setlh': 'setelah', 'seth': 'setelah', 'sllh': 'setelah', 'setelahh': 'setelah',
    'stlhh': 'setelah', 'setelaj': 'setelah', 'setelay': 'setelah', 'setellah': 'setelah', 'setla': 'setelah',
    'setlah': 'setelah', 'stla': 'setelah', 'stlah': 'setelah', 'stlhj': 'setelah', 'stlhy': 'setelah',
    'ssetelah': 'setelah', 'setelahh': 'setelah', 'setlla': 'setelah', 'setellahh': 'setelah', 'stlha': 'setelah',
    'setelaa': 'setelah', 'setelahj': 'setelah', 'setelahy': 'setelah',
    'after': 'setelah', 'aftr': 'setelah', 'aftur': 'setelah', 'afterr': 'setelah', 'afterj': 'setelah', 'aftery': 'setelah',
    # seusai
    'seusei': 'seusai', 'seusaii': 'seusai', 'susai': 'seusai', 'seusaj': 'seusai', 'seusay': 'seusai',
    'seusyy': 'seusai', 'seusjj': 'seusai', 'seusa': 'seusai',
    # selesai
    'slsai': 'selesai', 'selesei': 'selesai', 'selese': 'selesai', 'slesai': 'selesai', 'slese': 'selesai',
    'selesaii': 'selesai', 'slse': 'selesai', 'selesay': 'selesai', 'selesaj': 'selesai', 'sleesai': 'selesai',
    'slesa': 'selesai', 'selesaai': 'selesai', 'selesey': 'selesai', 'selesej': 'selesai', 'slsejj': 'selesai',
    'slseyy': 'selesai', 'slesei': 'selesai', 'slesey': 'selesai',
    'finish': 'selesai', 'fnish': 'selesai', 'finished': 'selesai', 'done': 'selesai', 'don': 'selesai',
    'donee': 'selesai', 'donej': 'selesai', 'doney': 'selesai',
    # sesudah
    'ssdh': 'sesudah', 'sesdh': 'sesudah', 'sesudahh': 'sesudah', 'ssdhh': 'sesudah', 'sesudaj': 'sesudah',
    'sesuday': 'sesudah', 'ssudah': 'sesudah', 'sesdah': 'sesudah', 'sesuda': 'sesudah', 'ssuda': 'sesudah',
    'sesudaah': 'sesudah', 'sesudahh': 'sesudah', 'sesudh': 'sesudah', 'sesudhh': 'sesudah', 'ssudaj': 'sesudah',
    'ssuday': 'sesudah',
    # selama
    'slma': 'selama', 'slama': 'selama', 'selamaa': 'selama', 'slamm': 'selama', 'selamaj': 'selama',
    'selamay': 'selama', 'selma': 'selama', 'selamm': 'selama', 'ssllama': 'selama', 'sellama': 'selama',
    'selamaaa': 'selama', 'slamaa': 'selama', 'slamaj': 'selama', 'slamay': 'selama',
    'during': 'selama', 'drg': 'selama', 'duringg': 'selama', 'duringj': 'selama', 'duringy': 'selama',
    # semenjak
    'smnjak': 'semenjak', 'semenjakk': 'semenjak', 'smnjk': 'semenjak', 'semenjakj': 'semenjak', 'semenjaky': 'semenjak',
    'sementak': 'semenjak', 'smenjakk': 'semenjak', 'semenjakkk': 'semenjak', 'smenjakj': 'semenjak', 'smenjaky': 'semenjak',
    # sejak
    'sjk': 'sejak', 'sejakk': 'sejak', 'sjkk': 'sejak', 'sejaj': 'sejak', 'sejay': 'sejak',
    'sjkj': 'sejak', 'sjky': 'sejak', 'sejkk': 'sejak', 'sjkkj': 'sejak', 'sjkky': 'sejak',
    'sjek': 'sejak', 'sjeck': 'sejak', 'ssejak': 'sejak', 'sejakj': 'sejak', 'sejaky': 'sejak',
    'since': 'sejak', 'snc': 'sejak', 'sincee': 'sejak', 'sincej': 'sejak', 'sincey': 'sejak',
    # hingga
    'hngga': 'hingga', 'hingg': 'hingga', 'hinga': 'hingga', 'hinggg': 'hingga', 'hinggaj': 'hingga',
    'hinggay': 'hingga', 'hngg': 'hingga', 'hnggaa': 'hingga', 'hinnga': 'hingga', 'hinggga': 'hingga',
    'hnggaj': 'hingga', 'hnggay': 'hingga',
    # sampai (sampe)
    'sampe': 'sampai', 'smpe': 'sampai', 'smpai': 'sampai', 'sampaii': 'sampai', 'sampey': 'sampai',
    'sampej': 'sampai', 'smpaj': 'sampai', 'smpay': 'sampai', 'sampaig': 'sampai', 'smpi': 'sampai',
    'smpej': 'sampai', 'smpey': 'sampai', 'smpaii': 'sampai', 'sampay': 'sampai', 'sampaay': 'sampai',
    'sampek': 'sampai', 'smpek': 'sampai', 'smpkk': 'sampai', 'sampekk': 'sampai', 'sampajj': 'sampai',
    'sampayy': 'sampai', 'smpekk': 'sampai', 'smpkkk': 'sampai',
    'until': 'sampai', 'till': 'sampai', 'untill': 'sampai', 'til': 'sampai', 'untilj': 'sampai',
    'untily': 'sampai', 'upto': 'sampai',
    # lalu
    'llu': 'lalu', 'laluu': 'lalu', 'laluj': 'lalu', 'laluy': 'lalu', 'lallu': 'lalu',
    'llo': 'lalu', 'lal': 'lalu', 'laluuu': 'lalu', 'lalujj': 'lalu', 'laluyy': 'lalu',
    'lluu': 'lalu',
    # kemudian
    'kmdian': 'kemudian', 'kmudian': 'kemudian', 'kmudn': 'kemudian', 'kemudiann': 'kemudian', 'kmdn': 'kemudian',
    'kemudiang': 'kemudian', 'kemudaj': 'kemudian', 'kemuday': 'kemudian', 'kkemudian': 'kemudian', 'kemudiann': 'kemudian',
    'kmduian': 'kemudian', 'kemudn': 'kemudian', 'kemudnn': 'kemudian', 'kemudaj': 'kemudian', 'kemuday': 'kemudian',
    'kmdiann': 'kemudian', 'kmudiann': 'kemudian',
    'then': 'kemudian', 'thn': 'kemudian', 'thenn': 'kemudian', 'thenj': 'kemudian', 'theny': 'kemudian',
    # terus
    'trs': 'terus', 'teruss': 'terus', 'trss': 'terus', 'trus': 'terus', 'truss': 'terus',
    'trusj': 'terus', 'trusy': 'terus', 'teros': 'terus', 'terosj': 'terus', 'terosy': 'terus',
    'terrus': 'terus', 'trrus': 'terus', 'tterus': 'terus', 'trussj': 'terus', 'trussy': 'terus',
    'next': 'terus', 'nxt': 'terus', 'nextt': 'terus', 'nextj': 'terus', 'nexty': 'terus',
    # sambil
    'smbil': 'sambil', 'sambl': 'sambil', 'sambill': 'sambil', 'sambilj': 'sambil', 'sambily': 'sambil',
    'smbl': 'sambil', 'samber': 'sambil', 'samberj': 'sambil', 'sambly': 'sambil', 'ssamibil': 'sambil',
    'samil': 'sambil', 'sambillj': 'sambil', 'sambilly': 'sambil',
    # sementara
    'smntara': 'sementara', 'smentara': 'sementara', 'smentra': 'sementara', 'semntara': 'sementara', 'smentaraa': 'sementara',
    'smentaraj': 'sementara', 'smentaray': 'sementara', 'smntr': 'sementara', 'smntrr': 'sementara', 'smentraaj': 'sementara',
    'smentraay': 'sementara', 'sementra': 'sementara',
    'meanwhile': 'sementara', 'mnwhile': 'sementara', 'meanwhilee': 'sementara', 'meanwhilej': 'sementara', 'meanwhiley': 'sementara',
    # tatkala
    'tatkalaa': 'tatkala', 'tatkalaj': 'tatkala', 'tatkalay': 'tatkala', 'ttkala': 'tatkala',
    'tatkl': 'tatkala', 'tatkla': 'tatkala', 'tatklaa': 'tatkala',
    # sewaktu
    'sewktu': 'sewaktu', 'sewaktuu': 'sewaktu', 'sewkt': 'sewaktu', 'sewaktuj': 'sewaktu', 'sewaktuy': 'sewaktu',
    'swaktu': 'sewaktu', 'sewakut': 'sewaktu', 'sewktjj': 'sewaktu', 'sewktyy': 'sewaktu',
    # seraya
    'sraya': 'seraya', 'serayaa': 'seraya', 'sry': 'seraya', 'serayaj': 'seraya', 'serayay': 'seraya',
    'srayaa': 'seraya', 'srayaj': 'seraya', 'srayay': 'seraya',
    # selagi
    'slagi': 'selagi', 'selagii': 'selagi', 'slgi': 'selagi', 'selagij': 'selagi', 'selagiy': 'selagi',
    'slagii': 'selagi', 'slagij': 'selagi', 'slagiy': 'selagi',
    # serta-merta (serta)
    'sertamerta': 'serta', 'sertamertaa': 'serta', 'srtamrta': 'serta', 'sertamertaj': 'serta', 'sertamertay': 'serta',
    # begitu
    'bgtu': 'begitu', 'bgitu': 'begitu', 'begtu': 'begitu', 'begituj': 'begitu', 'begituy': 'begitu',
    'bgjt': 'begitu', 'bgituh': 'begitu', 'begituh': 'begitu', 'bgtuu': 'begitu', 'bgituu': 'begitu',
    'bgtuj': 'begitu', 'bgtuy': 'begitu',
    'once': 'begitu', 'oncee': 'begitu', 'oncej': 'begitu', 'oncey': 'begitu',
    # seketika
    'sektika': 'seketika', 'seketikaa': 'seketika', 'sktika': 'seketika', 'seketikaj': 'seketika', 'seketikay': 'seketika',
    'seketikka': 'seketika', 'sktikaa': 'seketika',

    # ─── Kata Hubung, Preposisi & Penunjuk Umum ───
    # untuk
    'utk': 'untuk', 'untk': 'untuk', 'tuk': 'untuk', 'utuk': 'untuk', 'untukk': 'untuk',
    'untukj': 'untuk', 'untuky': 'untuk', 'utkk': 'untuk', 'untkks': 'untuk', 'utukj': 'untuk',
    'untukkk': 'untuk', 'u/': 'untuk', 'untuuk': 'untuk', 'untukkj': 'untuk', 'untunky': 'untuk',
    'utkkk': 'untuk', 'untkkk': 'untuk', 'untukkj': 'untuk', 'untukky': 'untuk', 'uutk': 'untuk',
    'untukkkj': 'untuk', 'untukkky': 'untuk',
    'for': 'untuk', 'forr': 'untuk', 'foor': 'untuk', 'fore': 'untuk', 'forj': 'untuk', 'fory': 'untuk',
    # dengan
    'dgn': 'dengan', 'dngn': 'dengan', 'dg': 'dengan', 'dgan': 'dengan', 'dng': 'dengan',
    'dgnn': 'dengan', 'dngnn': 'dengan', 'dengn': 'dengan', 'dengnn': 'dengan', 'dngan': 'dengan',
    'dngann': 'dengan', 'dgann': 'dengan', 'dengaj': 'dengan', 'dgj': 'dengan', 'dgy': 'dengan',
    'dngj': 'dengan', 'dngy': 'dengan', 'dngnnj': 'dengan', 'dengann': 'dengan', 'dngannn': 'dengan',
    'dgannn': 'dengan', 'degan': 'dengan', 'deganj': 'dengan', 'degany': 'dengan', 'dengannj': 'dengan',
    'denganny': 'dengan',
    'with': 'dengan', 'wth': 'dengan', 'wit': 'dengan', 'withh': 'dengan', 'withj': 'dengan', 'withy': 'dengan',
    # yang
    'yg': 'yang', 'yag': 'yang', 'iyg': 'yang', 'yangg': 'yang', 'ygg': 'yang',
    'yangj': 'yang', 'yangy': 'yang', 'ygn': 'yang', 'yanggg': 'yang', 'yagk': 'yang',
    'yank': 'yang', 'yq': 'yang', 'yggj': 'yang', 'yggy': 'yang', 'yanggk': 'yang',
    'yng': 'yang', 'yanggj': 'yang', 'yanggy': 'yang', 'ynag': 'yang', 'yan': 'yang',
    'yanh': 'yang',
    # karena
    'krn': 'karena', 'karna': 'karena', 'kerna': 'karena', 'krna': 'karena', 'karenaa': 'karena',
    'krnn': 'karena', 'karnaa': 'karena', 'krena': 'karena', 'krnaa': 'karena', 'krnna': 'karena',
    'krnny': 'karena', 'karnj': 'karena', 'karany': 'karena', 'karnej': 'karena', 'karney': 'karena',
    'krnnyy': 'karena', 'krnj': 'karena', 'krny': 'karena', 'karenaaa': 'karena', 'karenaaaj': 'karena',
    'karenaaay': 'karena', 'krnaaa': 'karena',
    'because': 'karena', 'coz': 'karena', 'cause': 'karena', 'cuz': 'karena', 'becoz': 'karena',
    'bcoz': 'karena', 'becuse': 'karena', 'becasue': 'karena', 'bcause': 'karena', 'becausej': 'karena',
    'becausey': 'karena',
    # dan
    'dn': 'dan', 'dann': 'dan', 'dannn': 'dan', 'danj': 'dan', 'dany': 'dan',
    'dj': 'dan', 'dnq': 'dan', 'danq': 'dan', 'dnn': 'dan', 'dnj': 'dan',
    'dny': 'dan', 'danjj': 'dan', 'danyy': 'dan', 'dnnn': 'dan', 'dnnnj': 'dan',
    'dnnny': 'dan',
    'and': 'dan', 'annd': 'dan', 'andd': 'dan', 'andj': 'dan', 'andy': 'dan',
    # atau
    'ato': 'atau', 'ataw': 'atau', 'ataupun': 'atau', 'atauu': 'atau', 'atou': 'atau',
    'atouu': 'atau', 'atawj': 'atau', 'atao': 'atau', 'ataou': 'atau', 'atuj': 'atau',
    'atuy': 'atau', 'atawy': 'atau', 'ataouu': 'atau', 'atapn': 'atau', 'atapun': 'atau',
    'ataupunn': 'atau', 'ataopun': 'atau',
    'or': 'atau', 'orr': 'atau', 'orj': 'atau', 'ory': 'atau',
    # dari
    'dr': 'dari', 'darii': 'dari', 'drr': 'dari', 'darij': 'dari', 'dariy': 'dari',
    'dary': 'dari', 'dry': 'dari', 'drrj': 'dari', 'drry': 'dari', 'dariii': 'dari',
    'drrr': 'dari', 'drrrj': 'dari', 'drrry': 'dari',
    'from': 'dari', 'frm': 'dari', 'fromm': 'dari', 'fromj': 'dari', 'fromy': 'dari',
    # di
    'dii': 'di', 'dj': 'di', 'dy': 'di', 'd': 'di',
    'djj': 'di', 'dyy': 'di', 'diii': 'di',
    'in': 'di', 'at': 'di', 'on': 'di',
    # ke
    'kee': 'ke', 'kj': 'ke', 'ky': 'ke', 'k': 'ke',
    'kjj': 'ke', 'kyy': 'ke', 'keee': 'ke',
    'to': 'ke', 'too': 'ke', 'toj': 'ke', 'toy': 'ke',
    # pada
    'pda': 'pada', 'pd': 'pada', 'padaa': 'pada', 'pdj': 'pada', 'pdy': 'pada',
    'pdaa': 'pada', 'pdd': 'pada', 'pdjj': 'pada', 'pdyy': 'pada', 'padaaa': 'pada',
    # dalam
    'kdalm': 'dalam', 'kedalm': 'dalam', 'dlm': 'dalam', 'dlam': 'dalam', 'dlmm': 'dalam',
    'dalamm': 'dalam', 'dalem': 'dalam', 'dlmj': 'dalam', 'dlmy': 'dalam', 'dalemj': 'dalam',
    'dalemy': 'dalam', 'dalamjj': 'dalam', 'dalamyy': 'dalam', 'dlmjj': 'dalam', 'dlmyy': 'dalam',
    'dalemm': 'dalam',
    # oleh
    'olh': 'oleh', 'olehh': 'oleh', 'olhh': 'oleh', 'olhy': 'oleh', 'olhj': 'oleh',
    'olhjj': 'oleh', 'olhyy': 'oleh', 'olehjj': 'oleh', 'olehyy': 'oleh',
    # serta
    'srta': 'serta', 'sertaa': 'serta', 'srt': 'serta', 'srtj': 'serta', 'srty': 'serta',
    'sertaj': 'serta', 'sertay': 'serta', 'srtaa': 'serta',
    # tetapi
    'tapi': 'tetapi', 'tp': 'tetapi', 'tpi': 'tetapi', 'tpii': 'tetapi', 'tpp': 'tetapi',
    'tapii': 'tetapi', 'tapij': 'tetapi', 'tapiy': 'tetapi', 'tapiij': 'tetapi', 'tapiiy': 'tetapi',
    'tapik': 'tetapi', 'tpjj': 'tetapi', 'tpyy': 'tetapi', 'tepy': 'tetapi', 'tetpi': 'tetapi',
    'tetpii': 'tetapi', 'tetpy': 'tetapi',
    'but': 'tetapi', 'butt': 'tetapi', 'butj': 'tetapi', 'buty': 'tetapi',
    # melainkan
    'mlainkn': 'melainkan', 'melainkn': 'melainkan', 'mlaenkn': 'melainkan', 'melainkann': 'melainkan', 'mlainknn': 'melainkan',
    'mlaenknn': 'melainkan', 'melainknj': 'melainkan', 'melainkny': 'melainkan',
    # sedangkan
    'sdgkn': 'sedangkan', 'sdngkn': 'sedangkan', 'sedgkn': 'sedangkan', 'sedngkan': 'sedangkan', 'sdgknn': 'sedangkan',
    'sdngknn': 'sedangkan', 'sedgknn': 'sedangkan', 'sedngkann': 'sedangkan', 'sedngkanj': 'sedangkan', 'sedngkany': 'sedangkan',
    # sehingga
    'shg': 'sehingga', 'sehinga': 'sehingga', 'shgg': 'sehingga', 'sehinggg': 'sehingga', 'shgjj': 'sehingga',
    'shgyy': 'sehingga', 'sehinggaj': 'sehingga', 'sehinggay': 'sehingga', 'sehingaaa': 'sehingga', 'sehingggj': 'sehingga',
    'sehingggy': 'sehingga',
    # maka
    'mka': 'maka', 'makaa': 'maka', 'mk': 'maka', 'mkj': 'maka', 'mky': 'maka',
    'makaaj': 'maka', 'makaay': 'maka', 'mkaa': 'maka', 'mkk': 'maka',
    # jika
    'jka': 'jika', 'jik': 'jika', 'jikj': 'jika', 'jiky': 'jika', 'jikk': 'jika',
    'kalu': 'jika', 'kalo': 'jika', 'kaloo': 'jika', 'klo': 'jika', 'kloo': 'jika',
    'klau': 'jika', 'klaau': 'jika', 'jikaj': 'jika', 'jikay': 'jika', 'kloj': 'jika',
    'kloy': 'jika', 'kaloj': 'jika', 'kaloy': 'jika', 'jikka': 'jika', 'jikaa': 'jika',
    'klooo': 'jika', 'klooj': 'jika', 'klooy': 'jika',
    'if': 'jika', 'iff': 'jika', 'ifj': 'jika', 'ify': 'jika',
    # agar
    'agr': 'agar', 'agarr': 'agar', 'spya': 'agar', 'spay': 'agar', 'supaya': 'agar',
    'spy': 'agar', 'agrj': 'agar', 'agry': 'agar', 'spyaj': 'agar', 'spyay': 'agar',
    'spyy': 'agar', 'agarjj': 'agar', 'agaryy': 'agar', 'supayaj': 'agar', 'supayay': 'agar',
    'supayya': 'agar',

    # ─── Kata Sifat, Suasana & Vibe Wisata ───
    # sejuk
    'sjuk': 'sejuk', 'sejukk': 'sejuk', 'sejok': 'sejuk', 'sejuky': 'sejuk', 'sejukj': 'sejuk',
    'sejuuk': 'sejuk', 'sjukk': 'sejuk', 'sjuuk': 'sejuk', 'sejukkk': 'sejuk', 'sjukj': 'sejuk',
    'sjuky': 'sejuk', 'sejokj': 'sejuk', 'sejoky': 'sejuk', 'ssejuk': 'sejuk', 'seejuk': 'sejuk',
    'sejukkj': 'sejuk', 'sejukky': 'sejuk',
    'cool': 'sejuk', 'cozy': 'sejuk', 'cold': 'sejuk', 'cozyy': 'sejuk', 'cozyj': 'sejuk',
    'cooll': 'sejuk', 'coolj': 'sejuk', 'cooly': 'sejuk', 'chill': 'sejuk', 'chilling': 'sejuk',
    'chilly': 'sejuk',
    # adem
    'adem': 'sejuk', 'ademm': 'sejuk', 'adhem': 'sejuk', 'adheml': 'sejuk', 'ademmm': 'sejuk',
    'ademy': 'sejuk', 'ademj': 'sejuk', 'adhemy': 'sejuk', 'adhemj': 'sejuk', 'ademyy': 'sejuk',
    'ademjj': 'sejuk', 'addeem': 'sejuk', 'addem': 'sejuk',
    # ramai
    'rame': 'ramai', 'ramee': 'ramai', 'rmei': 'ramai', 'bising': 'ramai', 'ramei': 'ramai',
    'ramaii': 'ramai', 'rameej': 'ramai', 'rameey': 'ramai', 'ramej': 'ramai', 'ramey': 'ramai',
    'ramay': 'ramai', 'ramajj': 'ramai', 'ramayy': 'ramai', 'rameee': 'ramai', 'rameeej': 'ramai',
    'rameeey': 'ramai', 'rammey': 'ramai', 'rammei': 'ramai', 'ramme': 'ramai',
    'crowded': 'ramai', 'crowd': 'ramai', 'crowdedd': 'ramai', 'crowdedj': 'ramai', 'crowdedy': 'ramai',
    'busy': 'ramai', 'busyy': 'ramai', 'busyj': 'ramai', 'noisy': 'ramai', 'noisyy': 'ramai',
    # sepi
    'spi': 'tenang', 'sepii': 'tenang', 'sepy': 'tenang', 'sunyi': 'tenang', 'sunyii': 'tenang',
    'sny': 'tenang', 'sepyy': 'tenang', 'sepij': 'tenang', 'sepiy': 'tenang', 'spiij': 'tenang',
    'spiiy': 'tenang', 'sssepi': 'tenang', 'ssepi': 'tenang', 'sepiii': 'tenang',
    # sunyi
    'snyi': 'tenang', 'suny': 'tenang', 'sunyj': 'tenang', 'sunyy': 'tenang', 'snyii': 'tenang',
    'sunyij': 'tenang', 'sunyiy': 'tenang', 'ssunyi': 'tenang', 'sunyiii': 'tenang',
    # damai
    'dmei': 'tenang', 'damei': 'tenang', 'damaii': 'tenang', 'dameey': 'tenang', 'damaj': 'tenang',
    'damay': 'tenang', 'damej': 'tenang', 'damey': 'tenang', 'damaij': 'tenang', 'damaiy': 'tenang',
    'dmaai': 'tenang', 'dmaaii': 'tenang',
    # tenang
    'tenag': 'tenang', 'tenangg': 'tenang', 'tng': 'tenang', 'tenangj': 'tenang', 'tenangy': 'tenang',
    'tnaang': 'tenang', 'tnaangg': 'tenang', 'tnang': 'tenang', 'tnangg': 'tenang', 'tenangjj': 'tenang',
    'tenangyy': 'tenang', 'ttenang': 'tenang', 'teenang': 'tenang', 'tenanng': 'tenang', 'tenanngg': 'tenang',
    'quiet': 'tenang', 'peaceful': 'tenang', 'calm': 'tenang', 'peace': 'tenang', 'silent': 'tenang',
    'quietj': 'tenang', 'quiety': 'tenang', 'peacefulj': 'tenang', 'peacefuly': 'tenang', 'calmj': 'tenang',
    'calmy': 'tenang',
    # indah
    'indh': 'indah', 'indahh': 'indah', 'indha': 'indah', 'indahy': 'indah', 'indahj': 'indah',
    'indhh': 'indah', 'pemandangan': 'indah', 'pmandangan': 'indah', 'pndangan': 'indah', 'indaah': 'indah',
    'indaahh': 'indah', 'indahjj': 'indah', 'indahyy': 'indah', 'iindah': 'indah', 'indhaaj': 'indah',
    'indhaay': 'indah',
    'beautiful': 'indah', 'beauty': 'indah', 'pretty': 'indah', 'gorgeous': 'indah', 'aesthetic': 'indah',
    'instagramable': 'indah', 'view': 'indah', 'scenery': 'indah', 'nice': 'indah', 'nicee': 'indah',
    'nicej': 'indah', 'nicey': 'indah', 'beautifulj': 'indah', 'beautifuly': 'indah', 'aestheticj': 'indah',
    'aestheticy': 'indah', 'instagramablej': 'indah', 'instagramabley': 'indah',
    # cantik
    'cntik': 'indah', 'cantikk': 'indah', 'cantj': 'indah', 'canty': 'indah', 'cntikk': 'indah',
    'cantikj': 'indah', 'cantiky': 'indah', 'cntikj': 'indah', 'cntiky': 'indah', 'ccantik': 'indah',
    'caantik': 'indah', 'cantikkj': 'indah', 'cantikky': 'indah',
    # bagus
    'bgs': 'bagus', 'baguss': 'bagus', 'bagos': 'bagus', 'bgus': 'bagus', 'bgss': 'bagus',
    'bagosj': 'bagus', 'bagosy': 'bagus', 'bagusj': 'bagus', 'bagusy': 'bagus', 'bbagus': 'bagus',
    'baagus': 'bagus', 'bagussj': 'bagus', 'bagussy': 'bagus', 'bgusj': 'bagus', 'bgusy': 'bagus',
    'great': 'bagus', 'awesome': 'bagus', 'excellent': 'bagus', 'amazing': 'bagus', 'perfect': 'bagus',
    'awesomej': 'bagus', 'awesomey': 'bagus', 'amazingj': 'bagus', 'amazingy': 'bagus', 'perfectj': 'bagus',
    'perfecty': 'bagus', 'good': 'bagus', 'goodd': 'bagus', 'goodj': 'bagus', 'goody': 'bagus',
    # keren
    'kerenn': 'bagus', 'kern': 'bagus', 'kerenj': 'bagus', 'kereny': 'bagus', 'kernn': 'bagus',
    'kerenjj': 'bagus', 'erenyy': 'bagus', 'kkeren': 'bagus', 'keeren': 'bagus', 'kerennj': 'bagus',
    'kerenny': 'bagus',
    # hebat
    'hbat': 'bagus', 'hebatt': 'bagus', 'hbt': 'bagus', 'hebatj': 'bagus', 'hebaty': 'bagus',
    'hbatt': 'bagus', 'hebatjj': 'bagus', 'hebatyy': 'bagus', 'hhebat': 'bagus', 'heebat': 'bagus',
    # seru
    'sru': 'senang', 'seruu': 'senang', 'seruj': 'senang', 'seruy': 'senang', 'sruj': 'senang',
    'sruy': 'senang', 'seruuu': 'senang', 'serujj': 'senang', 'seruyy': 'senang', 'sseru': 'senang',
    'seeruu': 'senang',
    # asik / asyik
    'asik': 'senang', 'asyik': 'senang', 'asikk': 'senang', 'asyikk': 'senang', 'ask': 'senang',
    'asikj': 'senang', 'asiky': 'senang', 'asyikj': 'senang', 'asyiky': 'senang', 'asikjj': 'senang',
    'asikyy': 'senang', 'aasik': 'senang', 'aasyik': 'senang',
    # nyaman
    'nyman': 'tenang', 'nyamann': 'tenang', 'nymnn': 'tenang', 'nyamanj': 'tenang', 'nyamany': 'tenang',
    'nyamannj': 'tenang', 'nyamanny': 'tenang', 'nnyaman': 'tenang', 'nyaaman': 'tenang', 'nymanj': 'tenang',
    'nymany': 'tenang',
    # bersih
    'brsih': 'bersih', 'bersihh': 'bersih', 'brsh': 'bersih', 'bersh': 'bersih', 'bersihj': 'bersih',
    'bersihy': 'bersih', 'brsihj': 'bersih', 'brsihy': 'bersih', 'bbersih': 'bersih', 'beersih': 'bersih',
    'bersihjj': 'bersih', 'bersihyy': 'bersih',
    # rapi
    'rpi': 'rapi', 'rapii': 'rapi', 'rapy': 'rapi', 'rapij': 'rapi', 'rapiy': 'rapi',
    'rrpi': 'rapi', 'rapiij': 'rapi', 'rapiiy': 'rapi', 'rrapi': 'rapi', 'raapii': 'rapi',
    # luas
    'lus': 'luas', 'luass': 'luas', 'luasss': 'luas', 'luss': 'luas', 'luasj': 'luas',
    'luasy': 'luas', 'luassj': 'luas', 'luassy': 'luas', 'lluas': 'luas', 'luaas': 'luas',
    'luassjj': 'luas', 'luassyy': 'luas',
    # senang
    'sng': 'senang', 'snenge': 'senang', 'hepi': 'senang', 'happy': 'senang', 'snang': 'senang',
    'hepii': 'senang', 'senangg': 'senang', 'senangj': 'senang', 'senangy': 'senang', 'sngg': 'senang',
    'senangjj': 'senang', 'senangyy': 'senang', 'ssenang': 'senang', 'seenang': 'senang', 'senanng': 'senang',
    'senanngg': 'senang', 'hepj': 'senang', 'hepy': 'senang',
    'happyj': 'senang', 'happyy': 'senang', 'excited': 'senang', 'enjoy': 'senang', 'enjoyable': 'senang',
    'fun': 'senang', 'funn': 'senang', 'funj': 'senang', 'funy': 'senang', 'joy': 'senang',
    # sedih
    'sedihh': 'sedih', 'sdeh': 'sedih', 'males': 'sedih', 'boring': 'sedih', 'bosen': 'sedih',
    'sedihy': 'sedih', 'sedihj': 'sedih', 'sdih': 'sedih', 'sdihh': 'sedih', 'sedihjj': 'sedih',
    'sedihyy': 'sedih', 'ssedih': 'sedih', 'seedih': 'sedih', 'sdihj': 'sedih', 'sdihy': 'sedih',
    'sad': 'sedih', 'bored': 'sedih', 'lonely': 'sedih', 'sadj': 'sedih', 'sady': 'sedih',
    # kuno
    'klasik': 'kuno', 'sejarah': 'sejarah', 'sejarahh': 'sejarah', 'sejrah': 'sejarah', 'purba': 'kuno',
    'kunoo': 'kuno', 'kunoj': 'kuno', 'kunoy': 'kuno', 'kunojj': 'kuno', 'kunoyy': 'kuno',
    'historical': 'sejarah', 'ancient': 'kuno', 'old': 'kuno', 'classic': 'kuno', 'antique': 'kuno',
    'heritage': 'sejarah',
    # estetis
    'aestetik': 'indah', 'estetik': 'indah', 'estetikk': 'indah', 'astetik': 'indah', 'astetikk': 'indah',
    'estetikj': 'indah', 'estetiky': 'indah', 'aestetikj': 'indah', 'aestetiky': 'indah', 'estetiik': 'indah',
    # mistis
    'mstis': 'mistis', 'mistiss': 'mistis', 'angker': 'mistis', 'wingit': 'mistis', 'mistisj': 'mistis',
    'mistisy': 'mistis', 'mstiss': 'mistis', 'mistissj': 'mistis', 'mistissy': 'mistis',
    'mystic': 'mistis', 'spooky': 'mistis', 'scary': 'mistis', 'haunted': 'mistis', 'horror': 'mistis',
    'creepy': 'mistis',
    # romantis
    'romantiss': 'romantis', 'romntis': 'romantis', 'pacaran': 'romantis', 'romantisj': 'romantis', 'romantisy': 'romantis',
    'romantissj': 'romantis', 'romantissy': 'romantis',
    'romantic': 'romantis', 'romance': 'romantis', 'romanticj': 'romantis', 'romanticy': 'romantis',
    # edukatif
    'edukasi': 'edukasi', 'edkasi': 'edukasi', 'pdidikan': 'edukasi', 'pendidikan': 'edukasi', 'belajr': 'edukasi',
    'belajar': 'edukasi', 'edukasij': 'edukasi', 'edukasiy': 'edukasi', 'edkasij': 'edukasi', 'edkasiy': 'edukasi',
    'belajarr': 'edukasi',
    'educational': 'edukasi', 'education': 'edukasi', 'learning': 'edukasi', 'study': 'edukasi',
    # modern
    'mdrn': 'modern', 'modernn': 'modern', 'modrn': 'modern', 'modernj': 'modern', 'moderny': 'modern',
    'modrnn': 'modern', 'mdrnn': 'modern', 'modren': 'modern', 'modrenj': 'modern', 'modreny': 'modern',
    'futuristic': 'modern', 'new': 'modern',
    # tradisional
    'tradsional': 'tradisional', 'tradisnal': 'tradisional', 'jadul': 'tradisional', 'tradisionalj': 'tradisional', 'tradisionaly': 'tradisional',
    'tradisonal': 'tradisional', 'tradisnalj': 'tradisional', 'tradisnaly': 'tradisional',
    'traditional': 'tradisional', 'local': 'tradisional', 'culture': 'tradisional', 'cultural': 'tradisional',
    # asri
    'asrii': 'asri', 'asry': 'asri', 'asrij': 'asri', 'asriy': 'asri', 'aasri': 'asri',
    'asriij': 'asri', 'asriiy': 'asri',
    # segar
    'sgar': 'segar', 'segarr': 'segar', 'seger': 'segar', 'segerr': 'segar', 'segarj': 'segar',
    'segary': 'segar', 'segerj': 'segar', 'segery': 'segar', 'ssegar': 'segar', 'seegar': 'segar',
    'segarrj': 'segar', 'segarry': 'segar',
    'fresh': 'segar', 'freshj': 'segar', 'freshy': 'segar', 'green': 'segar', 'greenj': 'segar',
    'greeny': 'segar',
    # murah
    'mrah': 'murah', 'murahh': 'murah', 'murahj': 'murah', 'murahy': 'murah', 'murahk': 'murah',
    'mraah': 'murah', 'murahjj': 'murah', 'murahyy': 'murah', 'mmurah': 'murah', 'muurah': 'murah',
    'murahjks': 'murah',
    'cheap': 'murah', 'budget': 'murah', 'affordable': 'murah', 'cheapj': 'murah', 'cheapy': 'murah',
    'budgetj': 'murah', 'budgety': 'murah',
    # mahal
    'mhal': 'mahal', 'mahall': 'mahal', 'mahalj': 'mahal', 'mahaly': 'mahal', 'mhaal': 'mahal',
    'mahaljj': 'mahal', 'mahalyy': 'mahal', 'mmahal': 'mahal', 'maahal': 'mahal', 'mahaljks': 'mahal',
    'expensive': 'mahal', 'pricey': 'mahal', 'costly': 'mahal', 'expensivej': 'mahal', 'expensivey': 'mahal',
    'priceyj': 'mahal', 'priceyy': 'mahal',
    # santai
    'sntai': 'santai', 'sante': 'santai', 'santaii': 'santai', 'santey': 'santai', 'santay': 'santai',
    'santaij': 'santai', 'santaiy': 'santai', 'santei': 'santai', 'sntaij': 'santai', 'sntaiy': 'santai',
    'santej': 'santai', 'santeyy': 'santai', 'sntaii': 'santai', 'sntay': 'santai',
    'relax': 'santai', 'relaxing': 'santai', 'relaxj': 'santai', 'relaxy': 'santai',

    # ─── Kata Kerja, Negasi & Kata Bantu ───
    # tidak
    'tdk': 'tidak', 'ttdk': 'tidak', 'tddk': 'tidak', 'tdkk': 'tidak', 'tidk': 'tidak',
    'tda': 'tidak', 'tada': 'tidak', 'tdax': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'ga': 'tidak', 'gak': 'tidak', 'gakk': 'tidak', 'gaa': 'tidak', 'g': 'tidak',
    'ndak': 'tidak', 'ndakk': 'tidak', 'nda': 'tidak', 'dak': 'tidak', 'dakk': 'tidak',
    'tk': 'tidak', 'tkk': 'tidak', 'tidakk': 'tidak', 'tidakj': 'tidak', 'tidaky': 'tidak',
    'gk': 'tidak', 'gkk': 'tidak', 'tdak': 'tidak', 'tida': 'tidak', 'tidaj': 'tidak',
    'tiday': 'tidak', 'ttdkkk': 'tidak', 'ttdkkj': 'tidak', 'ttdkky': 'tidak', 'tddkk': 'tidak',
    'tddkkj': 'tidak', 'tddkky': 'tidak', 'tdkkj': 'tidak', 'tdkky': 'tidak', 'tidkk': 'tidak',
    'tidkkj': 'tidak', 'tidkky': 'tidak', 'tidkks': 'tidak', 'tdkkks': 'tidak', 'tgak': 'tidak',
    'tggak': 'tidak',
    'no': 'tidak', 'not': 'tidak', 'dont': 'tidak', 'doesnt': 'tidak', 'never': 'tidak',
    'notj': 'tidak', 'noty': 'tidak', 'dontj': 'tidak', 'donty': 'tidak', 'noo': 'tidak',
    # jangan
    'jngn': 'tidak', 'jgn': 'tidak', 'jangann': 'tidak', 'jngnn': 'tidak', 'jgann': 'tidak',
    'janganj': 'tidak', 'jangany': 'tidak', 'jgannj': 'tidak', 'jganny': 'tidak', 'jgnj': 'tidak',
    'jgny': 'tidak', 'jngnj': 'tidak', 'jngny': 'tidak',
    # bukan
    'bkn': 'tidak', 'bukann': 'tidak', 'bknn': 'tidak', 'bukanj': 'tidak', 'bukany': 'tidak',
    'bknj': 'tidak', 'bkny': 'tidak', 'bukannj': 'tidak', 'bukanny': 'tidak',
    # bisa
    'bs': 'bisa', 'bisaa': 'bisa', 'bisaj': 'bisa', 'bisay': 'bisa', 'bss': 'bisa',
    'bisaas': 'bisa', 'bisajj': 'bisa', 'bisayy': 'bisa', 'bbisa': 'bisa', 'biisa': 'bisa',
    'bsj': 'bisa', 'bsy': 'bisa', 'bssj': 'bisa', 'bssy': 'bisa',
    'can': 'bisa', 'able': 'bisa', 'may': 'bisa', 'allow': 'bisa', 'canj': 'bisa', 'cany': 'bisa',
    # boleh
    'blh': 'bisa', 'bolehh': 'bisa', 'bolh': 'bisa', 'bolehj': 'bisa', 'bolehy': 'bisa',
    'blhj': 'bisa', 'blhy': 'bisa', 'bolhj': 'bisa', 'bolhy': 'bisa', 'bolehhh': 'bisa',
    'bboleh': 'bisa', 'booleh': 'bisa',
    # harus
    'hrs': 'harus', 'haruss': 'harus', 'hrss': 'harus', 'harusj': 'harus', 'harusy': 'harus',
    'hrsj': 'harus', 'hrsy': 'harus', 'hrssj': 'harus', 'hrssy': 'harus', 'hharus': 'harus',
    'haarus': 'harus', 'harussj': 'harus', 'harussy': 'harus',
    'must': 'harus', 'should': 'harus', 'mustj': 'harus', 'musty': 'harus', 'shouldj': 'harus',
    'shouldy': 'harus',
    # mau
    'mo': 'mau', 'mw': 'mau', 'mauu': 'mau', 'mauj': 'mau', 'mauy': 'mau',
    'mauuu': 'mau', 'maujj': 'mau', 'mauyy': 'mau', 'mmau': 'mau', 'maau': 'mau',
    'moj': 'mau', 'moy': 'mau', 'mwj': 'mau', 'mwy': 'mau',
    # ingin
    'ingn': 'ingin', 'pengen': 'ingin', 'pngen': 'ingin', 'pgen': 'ingin', 'pengin': 'ingin',
    'inginn': 'ingin', 'inginy': 'ingin', 'inginj': 'ingin', 'hendak': 'ingin', 'hndk': 'ingin',
    'inginnj': 'ingin', 'inginny': 'ingin', 'pengenn': 'ingin', 'pengenj': 'ingin', 'pengeny': 'ingin',
    'pngenj': 'ingin', 'pngeny': 'ingin', 'iingin': 'ingin', 'iingn': 'ingin',
    'want': 'ingin', 'wanna': 'ingin', 'wish': 'ingin', 'hope': 'ingin', 'wantj': 'ingin',
    'wanty': 'ingin', 'wannaj': 'ingin', 'wannay': 'ingin',
    # perlu
    'prlu': 'perlu', 'perluu': 'perlu', 'prl': 'perlu', 'perluj': 'perlu', 'perluy': 'perlu',
    'prluj': 'perlu', 'prluy': 'perlu', 'prllu': 'perlu', 'perllu': 'perlu', 'perluuu': 'perlu',
    'pperlu': 'perlu', 'peerlu': 'perlu',
    'need': 'perlu', 'require': 'perlu', 'needj': 'perlu', 'needy': 'perlu',
    # akan
    'akn': 'akan', 'akann': 'akan', 'bakalan': 'akan', 'bkal': 'akan', 'bkalan': 'akan',
    'akanj': 'akan', 'akany': 'akan', 'aknn': 'akan', 'akannj': 'akan', 'akanny': 'akan',
    'aakann': 'akan', 'aakan': 'akan', 'bkalann': 'akan', 'bkalj': 'akan', 'bkaly': 'akan',
    'will': 'akan', 'gonna': 'akan', 'willj': 'akan', 'willy': 'akan',
    # sudah
    'sdh': 'sudah', 'udah': 'sudah', 'udh': 'sudah', 'sudh': 'sudah', 'udhh': 'sudah',
    'sdhh': 'sudah', 'sda': 'sudah', 'uda': 'sudah', 'udaa': 'sudah', 'sudahj': 'sudah',
    'sudahy': 'sudah', 'udahj': 'sudah', 'udahy': 'sudah', 'udhjj': 'sudah', 'udhyy': 'sudah',
    'sdhjj': 'sudah', 'sdhyy': 'sudah', 'ssudah': 'sudah', 'ssudahj': 'sudah', 'ssudahy': 'sudah',
    'uudah': 'sudah', 'uudahj': 'sudah', 'uudahy': 'sudah', 'sdaah': 'sudah', 'sdaa': 'sudah',
    'already': 'sudah', 'alreadyj': 'sudah', 'alreadyy': 'sudah',
    # belum
    'blm': 'belum', 'belom': 'belum', 'blom': 'belum', 'belumm': 'belum', 'blmm': 'belum',
    'belumj': 'belum', 'belumy': 'belum', 'belomj': 'belum', 'belomy': 'belum', 'blomj': 'belum',
    'blomy': 'belum', 'blmjj': 'belum', 'blmyy': 'belum', 'bbelum': 'belum', 'beelum': 'belum',
    'belomm': 'belum', 'belommj': 'belum', 'belommy': 'belum',
    'notyet': 'belum', 'notyetj': 'belum', 'notyety': 'belum',
    # sedang
    'sdg': 'sedang', 'sdng': 'sedang', 'sedng': 'sedang', 'lagii': 'sedang', 'lagi': 'sedang',
    'lgi': 'sedang', 'sedangj': 'sedang', 'sedangy': 'sedang', 'sdgj': 'sedang', 'sdgy': 'sedang',
    'sdngj': 'sedang', 'sdngy': 'sedang', 'lagij': 'sedang', 'lagiy': 'sedang', 'ssedang': 'sedang',
    'seedang': 'sedang', 'sedangjj': 'sedang', 'sedangyy': 'sedang',
    'currently': 'sedang', 'now': 'sedang', 'noww': 'sedang', 'nowj': 'sedang', 'nowy': 'sedang',
    # tempat
    'tmpt': 'tempat', 'tpat': 'tempat', 'tempatj': 'tempat', 'tempaty': 'tempat', 'tempatt': 'tempat',
    'tmptt': 'tempat', 'tmptj': 'tempat', 'tmpty': 'tempat', 'tempattj': 'tempat', 'tempatty': 'tempat',
    'ttempat': 'tempat', 'teempat': 'tempat', 'tmppat': 'tempat', 'tmppatj': 'tempat', 'tmppaty': 'tempat',
    'place': 'tempat', 'spot': 'tempat', 'location': 'tempat', 'area': 'tempat', 'placej': 'tempat',
    'placey': 'tempat', 'spotj': 'tempat', 'spoty': 'tempat',
    # wisata
    'wst': 'wisata', 'wisataa': 'wisata', 'wisataj': 'wisata', 'wisatay': 'wisata', 'wisataaa': 'wisata',
    'wstt': 'wisata', 'rekreasi': 'wisata', 'rkreasi': 'wisata', 'lburan': 'wisata', 'liburan': 'wisata',
    'wisatajj': 'wisata', 'wisatayy': 'wisata', 'wwisata': 'wisata', 'wiisata': 'wisata', 'rekreass': 'wisata',
    'rekreassj': 'wisata', 'rekreassy': 'wisata', 'liburann': 'wisata', 'liburanj': 'wisata', 'liburany': 'wisata',
    'travel': 'wisata', 'trip': 'wisata', 'tourism': 'wisata', 'tourist': 'wisata', 'vacation': 'wisata',
    'holiday': 'wisata', 'tripj': 'wisata', 'tripy': 'wisata', 'holidayj': 'wisata', 'holidayy': 'wisata',
    # sangat
    'sngt': 'sangat', 'sngat': 'sangat', 'bgt': 'sangat', 'banget': 'sangat', 'bangett': 'sangat',
    'sangatj': 'sangat', 'sangaty': 'sangat', 'sangatt': 'sangat', 'sngtt': 'sangat', 'bget': 'sangat',
    'bangetss': 'sangat', 'sekali': 'sangat', 'skali': 'sangat', 'skli': 'sangat', 'sangatjj': 'sangat',
    'sangatyy': 'sangat', 'bgtj': 'sangat', 'bgty': 'sangat', 'bangetj': 'sangat', 'bangety': 'sangat',
    'sklij': 'sangat', 'skliy': 'sangat', 'ssangat': 'sangat', 'saaangat': 'sangat', 'bbanget': 'sangat',
    'baanget': 'sangat',
    'very': 'sangat', 'too': 'sangat', 'really': 'sangat', 'extremely': 'sangat', 'veryj': 'sangat',
    'veryy': 'sangat', 'reallyj': 'sangat', 'reallyy': 'sangat',
    # banyak
    'banyakj': 'banyak', 'banyakk': 'banyak', 'banyaj': 'banyak', 'banyakks': 'banyak', 'bnyak': 'banyak',
    'bnyakk': 'banyak', 'banyaky': 'banyak', 'banyk': 'banyak', 'banykk': 'banyak', 'banyac': 'banyak',
    'banyack': 'banyak', 'bnyk': 'banyak', 'bnykk': 'banyak', 'banyakkk': 'banyak', 'melimpah': 'banyak',
    'banyajk': 'banyak', 'banyaak': 'banyak', 'banyaakk': 'banyak', 'bnykkk': 'banyak', 'banykks': 'banyak',
    'bnyakj': 'banyak', 'bnyaky': 'banyak', 'banyakkks': 'banyak', 'banyakkky': 'banyak', 'banyakkkj': 'banyak',
    'bannyak': 'banyak', 'bannyakk': 'banyak', 'bannyaak': 'banyak', 'bannyaj': 'banyak',
    'many': 'banyak', 'much': 'banyak', 'alot': 'banyak', 'lots': 'banyak', 'manyj': 'banyak',
    'manyy': 'banyak', 'muchj': 'banyak', 'muchy': 'banyak',
    # sedikit
    'sdkit': 'sedikit', 'sdkt': 'sedikit', 'dikit': 'sedikit', 'dikitt': 'sedikit', 'dkit': 'sedikit',
    'sedikitj': 'sedikit', 'sedikity': 'sedikit', 'sdkitj': 'sedikit', 'sdkity': 'sedikit', 'sdktj': 'sedikit',
    'sdkty': 'sedikit', 'dikitj': 'sedikit', 'dikity': 'sedikit', 'ssedikit': 'sedikit', 'seedikit': 'sedikit',
    'dikittj': 'sedikit', 'dikitty': 'sedikit',
    'little': 'sedikit', 'few': 'sedikit', 'bit': 'sedikit', 'abit': 'sedikit', 'fewj': 'sedikit',
    'fewy': 'sedikit', 'littlej': 'sedikit', 'littley': 'sedikit',

    # ─── Objek & Destinasi Wisata ───
    # pantai
    'pntai': 'pantai', 'pante': 'pantai', 'pantaii': 'pantai', 'pantey': 'pantai', 'pnte': 'pantai',
    'pantay': 'pantai', 'pantaij': 'pantai', 'pantaiy': 'pantai', 'pntaii': 'pantai', 'pantej': 'pantai',
    'panteyj': 'pantai', 'pntej': 'pantai', 'pantayy': 'pantai', 'pantaijj': 'pantai', 'pantaiyy': 'pantai',
    'ppantai': 'pantai', 'paantai': 'pantai', 'pntay': 'pantai', 'pntey': 'pantai',
    'beach': 'pantai', 'coast': 'pantai', 'ocean': 'pantai', 'sea': 'pantai', 'beachj': 'pantai',
    'beachy': 'pantai', 'oceanj': 'pantai', 'oceany': 'pantai', 'beaches': 'pantai',
    # gunung
    'gnung': 'gunung', 'gunug': 'gunung', 'gunungg': 'gunung', 'gng': 'gunung', 'gunungj': 'gunung',
    'gunungy': 'gunung', 'gnng': 'gunung', 'gnungj': 'gunung', 'gnungy': 'gunung', 'gunungjj': 'gunung',
    'gunungyy': 'gunung', 'ggunung': 'gunung', 'guunung': 'gunung', 'gnngj': 'gunung', 'gnngy': 'gunung',
    'mountain': 'gunung', 'mount': 'gunung', 'hill': 'gunung', 'volcano': 'gunung', 'mountainj': 'gunung',
    'mountainy': 'gunung', 'mountj': 'gunung', 'mounty': 'gunung', 'hills': 'gunung',
    # hutan
    'htan': 'hutan', 'hutann': 'hutan', 'hutany': 'hutan', 'hutanj': 'hutan', 'rimba': 'hutan',
    'hutannj': 'hutan', 'hutanny': 'hutan', 'hhutan': 'hutan', 'huutan': 'hutan', 'rimbaa': 'hutan',
    'rimbaj': 'hutan', 'rimbay': 'hutan',
    'forest': 'hutan', 'jungle': 'hutan', 'woods': 'hutan', 'forestj': 'hutan', 'foresty': 'hutan',
    'junglej': 'hutan', 'jungley': 'hutan',
    # candi
    'cndi': 'candi', 'candii': 'candi', 'candij': 'candi', 'candiy': 'candi', 'kuil': 'candi',
    'candijj': 'candi', 'candiyy': 'candi', 'ccandi': 'candi', 'caandi': 'candi', 'candiij': 'candi',
    'candiiy': 'candi',
    'temple': 'candi', 'shrine': 'candi', 'templej': 'candi', 'templey': 'candi', 'temples': 'candi',
    # museum
    'msem': 'museum', 'museumm': 'museum', 'musem': 'museum', 'museumj': 'museum', 'museumy': 'museum',
    'musemj': 'museum', 'musemy': 'museum', 'mmuseum': 'museum', 'muuseum': 'museum', 'museuum': 'museum',
    'museuumj': 'museum', 'museuumy': 'museum',
    'museums': 'museum',
    # kuliner
    'kliner': 'kuliner', 'kulinerr': 'kuliner', 'kulner': 'kuliner', 'mkan': 'kuliner', 'makan': 'kuliner',
    'makann': 'kuliner', 'makanan': 'kuliner', 'kulinerj': 'kuliner', 'kulinery': 'kuliner', 'kulinerrj': 'kuliner',
    'kulinerry': 'kuliner', 'mkanan': 'kuliner', 'mkananj': 'kuliner', 'mkanany': 'kuliner', 'makannj': 'kuliner',
    'makanny': 'kuliner', 'kkuliner': 'kuliner', 'kuuliner': 'kuliner', 'mmakan': 'kuliner', 'maakan': 'kuliner',
    'food': 'kuliner', 'culinary': 'kuliner', 'eat': 'kuliner', 'eating': 'kuliner', 'foodj': 'kuliner',
    'foody': 'kuliner', 'culinaryj': 'kuliner', 'culinaryy': 'kuliner', 'foods': 'kuliner',
    # belanja
    'blanja': 'belanja', 'belanjaa': 'belanja', 'blanjaa': 'belanja', 'shopping': 'belanja', 'belanjaj': 'belanja',
    'belanjay': 'belanja', 'blanjaj': 'belanja', 'blanjay': 'belanja', 'shoppingj': 'belanja', 'shoppingy': 'belanja',
    'bbelanja': 'belanja', 'beelanja': 'belanja', 'bbelanjaa': 'belanja',
    'shop': 'belanja', 'buy': 'belanja', 'mall': 'belanja', 'shopj': 'belanja', 'shopy': 'belanja',
    # kota
    'kta': 'kota', 'kotaa': 'kota', 'kt': 'kota', 'kotaj': 'kota', 'kotay': 'kota',
    'ktaaj': 'kota', 'ktaay': 'kota', 'kkota': 'kota', 'koota': 'kota', 'kotaaa': 'kota',
    'kotajj': 'kota', 'kotayy': 'kota',
    'city': 'kota', 'town': 'kota', 'cityj': 'kota', 'cityy': 'kota', 'townj': 'kota',
    'towny': 'kota',
    # desa
    'dsa': 'desa', 'desaa': 'desa', 'desaj': 'desa', 'desay': 'desa', 'dsaaj': 'desa',
    'dsaay': 'desa', 'ddesa': 'desa', 'deesa': 'desa', 'desaaa': 'desa', 'desajj': 'desa',
    'desayy': 'desa',
    'village': 'desa', 'villagej': 'desa', 'villagey': 'desa', 'villages': 'desa',
    # alam
    'alm': 'alam', 'alamm': 'alam', 'alamj': 'alam', 'alamy': 'alam', 'alammj': 'alam',
    'alammy': 'alam', 'aalam': 'alam', 'alaam': 'alam', 'alamjj': 'alam', 'alamyy': 'alam',
    'nature': 'alam', 'natural': 'alam', 'naturej': 'alam', 'naturey': 'alam',
    # air / curug
    'ar': 'air', 'airr': 'air', 'curug': 'air', 'crg': 'air', 'airterjun': 'air',
    'airj': 'air', 'airy': 'air', 'airrj': 'air', 'airry': 'air', 'curugj': 'air',
    'curugy': 'air', 'crgj': 'air', 'crgy': 'air', 'aair': 'air', 'aiir': 'air',
    'curugg': 'air', 'curuggg': 'air',
    'water': 'air', 'waterfall': 'air', 'fall': 'air', 'falls': 'air', 'waterj': 'air',
    'watery': 'air', 'waterfallj': 'air', 'waterfally': 'air',
    # sungai
    'sngai': 'sungai', 'kali': 'sungai', 'sungaii': 'sungai', 'sungay': 'sungai', 'sungayj': 'sungai',
    'sungayy': 'sungai', 'kalij': 'sungai', 'kaliy': 'sungai', 'ssungai': 'sungai', 'suungai': 'sungai',
    'kalii': 'sungai', 'kaliij': 'sungai', 'kaliiy': 'sungai',
    'river': 'sungai', 'riverj': 'sungai', 'rivery': 'sungai', 'rivers': 'sungai',
    # danau
    'dnau': 'danau', 'danauu': 'danau', 'telaga': 'danau', 'danauj': 'danau', 'danauy': 'danau',
    'danauuj': 'danau', 'danauuy': 'danau', 'ddanau': 'danau', 'daanau': 'danau', 'telagaj': 'danau',
    'telagay': 'danau', 'telagaa': 'danau',
    'lake': 'danau', 'lakej': 'danau', 'lakey': 'danau', 'lakes': 'danau',
    # taman
    'tman': 'taman', 'tamann': 'taman', 'kebun': 'taman', 'kbun': 'taman', 'tamanj': 'taman',
    'tamany': 'taman', 'tamannj': 'taman', 'tamanny': 'taman', 'ttaman': 'taman', 'taaman': 'taman',
    'kebuny': 'taman', 'kbunj': 'taman', 'kbuny': 'taman',
    'park': 'taman', 'garden': 'taman', 'parkj': 'taman', 'parky': 'taman', 'gardenj': 'taman',
    'gardeny': 'taman', 'parks': 'taman',

    # ─── Waktu & Keadaan ───
    # hari
    'hri': 'hari', 'harii': 'hari', 'hr': 'hari', 'harij': 'hari', 'hariy': 'hari',
    'hrj': 'hari', 'hry': 'hari', 'hriij': 'hari', 'hriiy': 'hari', 'hhari': 'hari',
    'haari': 'hari', 'hariii': 'hari', 'hariijj': 'hari', 'hariiyy': 'hari',
    'day': 'hari', 'dayj': 'hari', 'dayy': 'hari', 'days': 'hari',
    # malam
    'mlm': 'malam', 'malamm': 'malam', 'malem': 'malam', 'malemm': 'malam', 'malamj': 'malam',
    'malamy': 'malam', 'malemj': 'malam', 'malemy': 'malam', 'malamjj': 'malam', 'malamyy': 'malam',
    'mmalam': 'malam', 'maalam': 'malam', 'malemmjj': 'malam', 'malemmyy': 'malam',
    'night': 'malam', 'nightj': 'malam', 'nighty': 'malam', 'nights': 'malam',
    # pagi
    'pgi': 'pagi', 'pagii': 'pagi', 'pagy': 'pagi', 'pagij': 'pagi', 'pagiy': 'pagi',
    'pgiij': 'pagi', 'pgiiy': 'pagi', 'ppagi': 'pagi', 'paagi': 'pagi', 'pagiij': 'pagi',
    'pagiiy': 'pagi', 'pagyy': 'pagi', 'pagyjj': 'pagi', 'pagyyy': 'pagi',
    'morning': 'pagi', 'morningj': 'pagi', 'morningy': 'pagi',
    # siang
    'sing': 'siang', 'siangg': 'siang', 'syang': 'siang', 'siangj': 'siang', 'siangy': 'siang',
    'singj': 'siang', 'singy': 'siang', 'sianggj': 'siang', 'sianggy': 'siang', 'ssiang': 'siang',
    'siiaang': 'siang', 'siangjj': 'siang', 'siangyy': 'siang',
    'afternoon': 'siang', 'noon': 'siang', 'afternoonj': 'siang', 'afternoony': 'siang',
    # sore
    'sre': 'sore', 'soree': 'sore', 'sory': 'sore', 'sorej': 'sore', 'sorey': 'sore',
    'srej': 'sore', 'srey': 'sore', 'soreej': 'sore', 'soreey': 'sore', 'ssore': 'sore',
    'soore': 'sore', 'soreejj': 'sore', 'soreeyy': 'sore',
    'evening': 'sore', 'eveningj': 'sore', 'eveningy': 'sore',
    # waktu
    'wktu': 'waktu', 'waktuu': 'waktu', 'waktij': 'waktu', 'waktiy': 'waktu', 'wkt': 'waktu',
    'waktuj': 'waktu', 'waktuy': 'waktu', 'wwaktu': 'waktu', 'waaktu': 'waktu', 'wktuj': 'waktu',
    'wktuy': 'waktu',
    'time': 'waktu', 'timej': 'waktu', 'timey': 'waktu', 'times': 'waktu',
    # jam
    'jm': 'jam', 'jamm': 'jam', 'jamj': 'jam', 'jamy': 'jam', 'jammj': 'jam',
    'jammy': 'jam', 'jjm': 'jam', 'jaam': 'jam', 'jamjj': 'jam', 'jamyy': 'jam',
    'hour': 'jam', 'hours': 'jam', 'hourj': 'jam', 'houry': 'jam',

    # ─── Pelaku & Sosial ───
    # saya / aku
    'sya': 'saya', 'sayaa': 'saya', 'aku': 'saya', 'ak': 'saya', 'akk': 'saya',
    'gw': 'saya', 'gue': 'saya', 'gwe': 'saya', 'sayaj': 'saya', 'sayay': 'saya',
    'akuj': 'saya', 'akuy': 'saya', 'gwej': 'saya', 'gwey': 'saya', 'guej': 'saya',
    'guey': 'saya', 'ssaya': 'saya', 'saaya': 'saya', 'aaku': 'saya', 'gww': 'saya',
    'gwee': 'saya', 'gwejj': 'saya', 'gweyy': 'saya',
    'me': 'saya', 'my': 'saya', 'self': 'saya',
    # kamu
    'kmu': 'kamu', 'lu': 'kamu', 'lo': 'kamu', 'kmuu': 'kamu', 'kamuj': 'kamu',
    'kamuy': 'kamu', 'luj': 'kamu', 'luy': 'kamu', 'loj': 'kamu', 'loy': 'kamu',
    'kmuuj': 'kamu', 'kmuuy': 'kamu', 'kkamu': 'kamu', 'kaamu': 'kamu', 'luu': 'kamu',
    'loo': 'kamu', 'luuj': 'kamu', 'luuy': 'kamu', 'looj': 'kamu', 'looy': 'kamu',
    'you': 'kamu', 'your': 'kamu', 'u': 'kamu',
    # kita
    'kta': 'kita', 'kitaa': 'kita', 'kt': 'kita', 'kitaj': 'kita', 'kitay': 'kita',
    'ktaj': 'kita', 'ktay': 'kita', 'kkita': 'kita', 'kiita': 'kita', 'kitaaa': 'kita',
    'kitajj': 'kita', 'kitayy': 'kita',
    # kami
    'kmi': 'kami', 'kamii': 'kami', 'kamij': 'kami', 'kamiy': 'kami', 'kmij': 'kami',
    'kmiy': 'kami', 'kkami': 'kami', 'kaami': 'kami', 'kamiij': 'kami', 'kamiiy': 'kami',
    # mereka
    'mrka': 'mereka', 'mrek': 'mereka', 'merekaj': 'mereka', 'merekay': 'mereka', 'mraj': 'mereka',
    'mray': 'mereka', 'mmereka': 'mereka', 'meereka': 'mereka', 'merekk': 'mereka', 'merekkj': 'mereka',
    'merekky': 'mereka',
    # keluarga
    'klurga': 'keluarga', 'keluargaa': 'keluarga', 'klg': 'keluarga', 'kelg': 'keluarga',
    'family': 'keluarga', 'keluargaj': 'keluarga', 'keluargay': 'keluarga', 'klgj': 'keluarga', 'klgy': 'keluarga',
    'kelgj': 'keluarga', 'kelgy': 'keluarga', 'kkeluarga': 'keluarga', 'keeluarga': 'keluarga', 'keluargajj': 'keluarga',
    'keluargayy': 'keluarga',
    # teman
    'tman': 'teman', 'temann': 'teman', 'tmn': 'teman', 'tmn-tmn': 'teman', 'tmen': 'teman',
    'sahabat': 'teman', 'temanj': 'teman', 'temany': 'teman', 'tmnj': 'teman', 'tmny': 'teman',
    'tmenj': 'teman', 'tmeny': 'teman', 'tteman': 'teman', 'teeman': 'teman', 'temannj': 'teman',
    'temanny': 'teman', 'sahabatt': 'teman', 'sahabatj': 'teman', 'sahabaty': 'teman',
    # pasangan
    'psangan': 'pasangan', 'pasangann': 'pasangan', 'pacar': 'pasangan', 'pacarr': 'pasangan',
    'doi': 'pasangan', 'doii': 'pasangan', 'pasanganj': 'pasangan', 'pasangany': 'pasangan', 'pacarj': 'pasangan',
    'pacary': 'pasangan', 'doij': 'pasangan', 'doiy': 'pasangan', 'ppasangan': 'pasangan', 'paasangan': 'pasangan',
    'pacarrj': 'pasangan', 'pacarry': 'pasangan', 'doiii': 'pasangan', 'doijjj': 'pasangan', 'doiyyy': 'pasangan',
    # camping
    'kemping': 'camping', 'campingg': 'camping', 'camp': 'camping', 'campingan': 'camping',
    'berkemah': 'camping', 'campingj': 'camping', 'campingy': 'camping', 'kempingj': 'camping', 'kempingy': 'camping',
    'campj': 'camping', 'campy': 'camping', 'berkemahj': 'camping', 'berkemahy': 'camping', 'ccamping': 'camping',
    'caamping': 'camping', 'kempingg': 'camping',
    # main
    'men': 'main', 'mainn': 'main', 'maen': 'main', 'maenn': 'main', 'mainj': 'main',
    'mainy': 'main', 'maenj': 'main', 'maeny': 'main', 'mainjj': 'main', 'mainyy': 'main',
    'mmain': 'main', 'maain': 'main', 'maennj': 'main', 'maenny': 'main',
    # lihat
    'lhat': 'lihat', 'liat': 'lihat', 'liatt': 'lihat', 'lihhat': 'lihat', 'lihatj': 'lihat',
    'lihaty': 'lihat', 'liatj': 'lihat', 'liaty': 'lihat', 'liatjj': 'lihat', 'liatyy': 'lihat',
    'llihat': 'lihat', 'liihat': 'lihat', 'lihatjj': 'lihat', 'lihatyy': 'lihat',
    # foto
    'poto': 'foto', 'photo': 'foto', 'potoo': 'foto', 'fotoo': 'foto', 'selfie': 'foto',
    'fotoj': 'foto', 'fotoy': 'foto', 'potoj': 'foto', 'potoy': 'foto', 'photoj': 'foto',
    'photoy': 'foto', 'ffoto': 'foto', 'ffotoo': 'foto', 'ppoto': 'foto', 'ppotoo': 'foto',
    'photooj': 'foto', 'photooy': 'foto',
    # anak
    'ank': 'anak', 'anakk': 'anak', 'bocah': 'anak', 'anak-anak': 'anak', 'anakj': 'anak',
    'anaky': 'anak', 'anakkj': 'anak', 'anakky': 'anak', 'bocahj': 'anak', 'bocahy': 'anak',
    'aanak': 'anak', 'anakkk': 'anak', 'anakkkjj': 'anak', 'anakkkyy': 'anak',
    # orang tua
    'orangtua': 'orang tua', 'ortu': 'orang tua', 'bapak': 'orang tua', 'ibu': 'orang tua',
    'orangtuaj': 'orang tua', 'orangtuay': 'orang tua', 'ortuj': 'orang tua', 'ortuy': 'orang tua', 'bapakj': 'orang tua',
    'bapaky': 'orang tua', 'ibuj': 'orang tua', 'ibuy': 'orang tua', 'orangtuaa': 'orang tua', 'ortuu': 'orang tua'
}

# Regex list to detect and strip potential adversarial prompt injections/jailbreaks
INJECTION_PATTERNS = [
    r'lupakan\s+(instruksi|perintah|arahan)(?:\s+sebelumnya)?',
    r'ignore\s+(instructions|previous|commands|prompt|rules)(?:\s+above)?',
    r'override\s+(prompt|system|instructions)',
    r'system\s+prompt',
    r'bertindaklah\s+sebagai',
    r'act\s+as\s+a',
]

def clean_and_correct_text(text):
    """
    Cleans the input text by:
    1. Removing adversarial prompt injection attempts without crashing the request.
    2. Lowercasing & trimming.
    3. Removing HTML tags and safe regex sanitation against script/SQL injections.
    4. Normalizing common typos and informal Indonesian words.
    """
    if not text:
        return ""
    
    # 1. Strip adversarial injection words/phrases first (case-insensitive)
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    
    # 2. Lowercase & strip
    cleaned = cleaned.lower().strip()
    
    # 3. Sanitize: remove HTML tags/special characters to prevent any prompt injection attempts
    cleaned = re.sub(r'<[^>]*>', ' ', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
    
    # 4. Simple spell correction / normalization for typos
    words = cleaned.split()
    corrected_words = [INDONESIAN_SPELL_CORRECTION.get(w, w) for w in words]
    
    return " ".join(corrected_words)

CATEGORY_MAPPING = {
    'Alam': ['Alam', 'Pantai', 'Wisata Air'],
    'Budaya_Dan_Sejarah': ['Budaya_Dan_Sejarah', 'Museum', 'Religi', 'Seni'],
    'Edukasi': ['Pendidikan', 'Museum', 'Desa Wisata', 'Agrowisata'],
    'Taman_Hiburan': ['Buatan', 'Minat Khusus'],
    'Pusat_Perbelanjaan': ['Kuliner', 'Buatan']
}

BACKWARD_MAPPING = {
    'Alam': 'Alam',
    'Pantai': 'Alam',
    'Wisata Air': 'Alam',
    'Budaya_Dan_Sejarah': 'Budaya_Dan_Sejarah',
    'Museum': 'Budaya_Dan_Sejarah',
    'Religi': 'Budaya_Dan_Sejarah',
    'Seni': 'Budaya_Dan_Sejarah',
    'Pendidikan': 'Edukasi',
    'Desa Wisata': 'Edukasi',
    'Agrowisata': 'Edukasi',
    'Buatan': 'Taman_Hiburan',
    'Minat Khusus': 'Taman_Hiburan',
    'Kuliner': 'Pusat_Perbelanjaan'
}

DEFAULT_IMAGES = {
    'Alam': 'https://res.cloudinary.com/dfciqrwpe/image/upload/v1748938869/tebing-breksi_ydjncv.jpg',
    'Budaya_Dan_Sejarah': 'https://res.cloudinary.com/dfciqrwpe/image/upload/v1748938870/candi-prambanan_cpukic.jpg',
    'Edukasi': 'https://res.cloudinary.com/dfciqrwpe/image/upload/v1748938869/ullen-sentalu_wvqzjf.jpg',
    'Taman_Hiburan': 'https://res.cloudinary.com/dfciqrwpe/image/upload/v1748938869/taman-sari_xpkjzl.jpg',
    'Pusat_Perbelanjaan': 'https://images.unsplash.com/photo-1555529669-e69e7aa0ba9a?auto=format&fit=crop&w=800&q=80'
}

def get_client_ip(request):
    """
    Safely extract client IP from incoming request, handling proxy headers.
    """
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

@csrf_exempt
@LATENCY_HISTOGRAM.time()
def get_itinerary(request):
    """
    Generate customized itinerary based on user preferences and content-based similarity.
    Includes in-memory sentiment inference on the user's active mood/vibe.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)
        
    try:
        # DDoS & Rate Limiting Protection (Max 30 requests per minute per IP)
        ip = get_client_ip(request)
        rate_key = f"rate_limit_{ip}"
        req_count = cache.get(rate_key, 0)
        
        if req_count >= 30:
            return JsonResponse({
                'status': 'error',
                'message': 'Batas limit kueri terlampaui. Harap tunggu 1 menit sebelum mencoba lagi.'
            }, status=429)
            
        cache.set(rate_key, req_count + 1, timeout=60)

        data = json.loads(request.body)
        
        # 0. Input Validation & Sabotage Protection
        minat = data.get('minat', [])
        if not isinstance(minat, list):
            return JsonResponse({'status': 'error', 'message': 'Minat harus berupa list/array'}, status=400)
            
        # Limit minat items to valid categories
        minat = [m for m in minat if m in CATEGORY_MAPPING]

        # Restrict budget
        try:
            budget = int(data.get('budget', 50000))
            if budget < 0:
                budget = 0
            elif budget > 10_000_000: # Capped at 10 million IDR to prevent integer overflow/sabotage
                budget = 10_000_000
        except (ValueError, TypeError):
            budget = 50000

        # Restrict durasi to 1 - 7 days to match frontend UX and prevent heavy loop DoS
        try:
            durasi = int(data.get('durasi', 3))
            if durasi < 1:
                durasi = 1
            elif durasi > 7:
                durasi = 7
        except (ValueError, TypeError):
            durasi = 3

        # Validate and clean suasana string
        raw_suasana = str(data.get('suasana', ''))
        if len(raw_suasana) > 500: # Cap string length to 500 chars to avoid vectorizer overloading
            raw_suasana = raw_suasana[:500]
            
        suasana = clean_and_correct_text(raw_suasana)
        
        # Load ML components from AppConfig
        api_app = apps.get_app_config('api')
        tfidf = api_app.ml_models['tfidf']
        logreg = api_app.ml_models['logreg']
        le = api_app.ml_models['le']
        cosine_sim = api_app.ml_models['cosine_sim']
        df_wisata = api_app.ml_models['df_wisata']
        name_to_idx = api_app.ml_models['name_to_idx']
        
        # 1. Run Sentiment Analysis if mood/vibe is specified
        sentiment_label = "Positif" # default
        if suasana.strip():
            X_text = tfidf.transform([suasana])
            pred_idx = logreg.predict(X_text)[0]
            sentiment_label = le.inverse_transform([pred_idx])[0]
            # Record Prometheus metric
            SENTIMENT_COUNTER.labels(sentiment_label=sentiment_label).inc()
        else:
            SENTIMENT_COUNTER.labels(sentiment_label='None').inc()
            
        # 2. Filter dataset based on category interests
        target_categories = []
        for m in minat:
            if m in CATEGORY_MAPPING:
                target_categories.extend(CATEGORY_MAPPING[m])
                
        # If no matching categories or empty minat, match all categories
        if not target_categories:
            target_categories = list(df_wisata['type'].unique())
            
        # Filter candidates by category and budget (using a flexible margin if too strict)
        candidates = df_wisata[df_wisata['type'].isin(target_categories)]
        
        budget_candidates = candidates[candidates['htm_weekday'] <= budget]
        if budget_candidates.empty:
            # Relax budget constraint if no candidates fit the budget
            budget_candidates = candidates[candidates['htm_weekday'] <= budget * 2.5]
            
        if budget_candidates.empty:
            # Fall back to all data if still empty
            budget_candidates = df_wisata
            
        # 3. CBF Similarity-Chain Itinerary Builder
        itinerary = {}
        visited = set()
        
        has_suasana = bool(suasana.strip())
        if has_suasana:
            query_vector = tfidf.transform([suasana])
            
        # Dynamic spot count per day based on matching candidate density and duration
        # We want to give as many relevant spots as possible, within a physically realistic range of 3 to 6 spots/day.
        try:
            n_candidates = len(budget_candidates)
            spots_per_day = max(3, min(6, n_candidates // durasi))
        except ZeroDivisionError:
            spots_per_day = 4
        except Exception:
            spots_per_day = 4

        for day in range(1, durasi + 1):
            day_label = f"Hari {day}"
            day_spots = []
            
            # Dynamically loops over the calculated spots_per_day
            for spot_idx in range(spots_per_day):
                # Backup logic: check if budget_candidates is empty and recover gracefully
                if budget_candidates.empty:
                    budget_candidates = df_wisata
                    
                available = budget_candidates[~budget_candidates['nama'].isin(visited)]
                if available.empty:
                    # Reset visited if we ran out of unique spots in the candidate pool
                    visited.clear()
                    available = budget_candidates
                    
                # Extra safety backup fallback: if available is still empty, force fallback to df_wisata
                if available.empty:
                    available = df_wisata
                    
                if spot_idx == 0:
                    if has_suasana:
                        # Vectorize currently available candidate descriptions
                        candidate_desc = available['description'].fillna('').tolist()
                        candidate_vectors = tfidf.transform(candidate_desc)
                        
                        # Compute exact cosine similarity (dot product of L2-normalized vectors)
                        scores = (query_vector * candidate_vectors.T).toarray()[0]
                        
                        # Find the candidate with the highest similarity score
                        best_score_idx = np.argmax(scores)
                        selected_spot = available.iloc[best_score_idx]
                    else:
                        # Pick the highest-rated spot matching categories as the seed for the day
                        selected_spot = available.sort_values(by='vote_average', ascending=False).iloc[0]
                else:
                    # Use Content-Based Similarity chain:
                    # Find available spot with highest cosine similarity to the last selected spot
                    last_spot_name = day_spots[-1]['nama']
                    last_idx = name_to_idx.get(last_spot_name)
                    
                    if last_idx is not None and last_idx < len(cosine_sim):
                        # Get similarity scores for last spot
                        sim_scores = cosine_sim[last_idx]
                        
                        # Find closest available spot
                        best_sim = -1
                        selected_spot = None
                        for _, row in available.iterrows():
                            row_name = row['nama']
                            row_orig_idx = name_to_idx.get(row_name)
                            if row_orig_idx is not None and row_orig_idx < len(sim_scores):
                                score = sim_scores[row_orig_idx]
                                if score > best_sim:
                                    best_sim = score
                                    selected_spot = row
                                    
                        if selected_spot is None:
                            selected_spot = available.sort_values(by='vote_average', ascending=False).iloc[0]
                    else:
                        selected_spot = available.sort_values(by='vote_average', ascending=False).iloc[0]
                        
                # Add to visited
                visited.add(selected_spot['nama'])
                
                # Format spot fields matching frontend
                frontend_type = BACKWARD_MAPPING.get(selected_spot['type'], 'Alam')
                image_url = selected_spot['image'] if isinstance(selected_spot['image'], str) and selected_spot['image'].startswith('http') else DEFAULT_IMAGES.get(frontend_type, DEFAULT_IMAGES['Alam'])
                
                day_spots.append({
                    'nama': selected_spot['nama'],
                    'type': frontend_type,
                    'vote_average': float(selected_spot['vote_average']),
                    'htm_weekday': int(selected_spot['htm_weekday']),
                    'image': image_url,
                    'description': selected_spot['description'] if isinstance(selected_spot['description'], str) else "Nikmati pemandangan indah di objek wisata pilihan terfavorit di Yogyakarta.",
                    'sentiment_label': sentiment_label
                })
                
            itinerary[day_label] = day_spots
            
        # Calculate total cost
        total_cost = sum(spot['htm_weekday'] for spots in itinerary.values() for spot in spots)
        
        # Record Prometheus metric
        ITINERARY_COUNTER.inc()
        
        return JsonResponse({
            'status': 'success',
            'estimasi_total_tiket': total_cost,
            'itinerary': itinerary
        })
        
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
