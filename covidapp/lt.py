import psycopg2
from django.template import loader
from django.http import HttpResponse
from .models import uji
conn = psycopg2.connect(host="localhost",database="covapp",port="5432",user="postgres",password="0000")   
cur = conn.cursor()

def delete(request):
    commandd = "Select tweet_id from covidapp_uji"
    cur.execute(commandd)
    rows2 = cur.fetchall()

    command = "Select tweet_id from covidapp_latih"
    cur.execute(command)
    rows = cur.fetchall()

    commands = "Select * from covidapp_uji"
    cur.execute(commands)
    rows3 = cur.fetchall() 
    def ins(u):
        for i in rows3:
            tweet_id = i[0]
            tweet_text = i[1]
            tgl = i[2]
            label = i[3]
            if tweet_id == u:
                commande = "INSERT INTO covidapp_latih VALUES (%s,%s,%s,%s);"
                cur.execute(commande,(tweet_id, tweet_text, tgl, label))
                conn.commit()
                commands = "delete from covidapp_uji where tweet_id = %s;"
                cur.execute(commands,(tweet_id,))
                conn.commit()
                commandt = "delete from covidapp_sentimen where tweet_id = %s;"
                cur.execute(commandt,(tweet_id,))
                conn.commit()
                commandp = "delete from covidapp_stemmed where tweet_id = %s;"
                cur.execute(commandp,(tweet_id,))
                conn.commit()
            else:
                pass
                
    for i in rows2:
        if i in rows:
            pass
        elif i not in rows:
            m = i[0]
            ins(m)
            
        else :
            pass

    golek = uji.objects.all()
    template = loader.get_template('covidapp/lu.html')
    jml = len(golek)
    ent = 'Data Uji'
    context = {
        'form': golek,
        'leng': jml,
        'enti' : ent,
    }
    return HttpResponse(template.render(context, request))