from flask import Flask
from flask import render_template
import sqlite3
import ffmpeg_streaming
from ffmpeg_streaming import Formats
import json

app = Flask(__name__)

# with open('data.json', 'r') as f:
#     repo = json.load(f)
# columns = ['id', 'title', 'release_date', 'cover', 'director', 'rating', 'synopsis', 'url']
#
# conn = sqlite3.connect('db.sqlite')
# cursor = conn.cursor()
# cursor.execute('CREATE TABLE if not exists Movie(id INTEGER, title TEXT, release_date TEXT, cover TEXT, '
#                'director TEXT, rating TEXT, synopsis TEXT, url TEXT)')
# for index in repo:
#     values = tuple(repo.get(index).get(c) for c in columns)
#     cursor.execute('INSERT INTO Movie values (?,?,?,?,?,?,?,?)', values)
# conn.commit()
# conn.close()


def fetch(_id):
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Movie WHERE id = ' + str(_id))
    res = cursor.fetchall()
    return res


def fetch_all():
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Movie')
    res = cursor.fetchall()
    return res


@app.route('/')
@app.route('/index')
def home():
    repo = fetch_all()
    return render_template('index.html', content=repo)


@app.route('/movies/<int:_id>')
def movie(_id):
    repo = fetch(_id)[0]
    return render_template('movie.html', content=repo)


if __name__ == '__main__':
    app.run()
