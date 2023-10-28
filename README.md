# elar.ink
elar.ink is a free and open source web application for English to Kannada dictionary. The project is inspired by Kannada-English dictionary [alar.ink](https://alar.ink/). The elar.ink is built on top of [dictpress](https://github.com/knadh/dictpress), [alar data](https://github.com/alar-dict/data), and [alar.ink website theme](https://github.com/alar-dict/alar.ink).
Currently, the dictionary is hosted at [dictionary.decodingbackend.com](https://dictionary.decodingbackend.com/) but I plan to create a separate domain name for it in the future.

The alar data contains Kannada word to English definitions. 
```yaml
- id: 59 # Orignal ID of the entry
  head: ಅ # Alphabet / first letter of the entry word.
  entry: ಅಂಕಪರದೆ # Entry Word.
  phone: aŋka parade # Phonetic notation.
  origin: ""
  info: ""
  defs:
  - id: 335427
    entry: the curtain used to pull down at the end of a scene or act, in a play.
    type: noun
  - id: 211623
    entry: (fig.) the end of or an action bringing an end to, an event or an occasion;
    type: noun
  - id: 336691
    entry: ಅಂಕಪರದೆಯೇಳು aŋkaparadeyēḷu (the performance of a dramatic act) to start (as on the stage).
    type: noun
  - id: 237657
    entry: (fig.) (a new phase of action, style, etc.) to commence; ಅಂಕಪರದೆಬೀಳು aŋkaparade bīḷu to come to an end; 2. to cause to end.
    type: noun
```

To obtain relevant Kannada words for a given English word, we embed the search word, then run the similarity search against all the English definitions, sort the results by relevance and return the Kannada-English definition pairs.

## Installation
After cloning the project, create a virtual environment and install the requirements.
```bash
python -m virtualenv venv # tested on Python3.8+
source venv/bin/activate

pip install -r requirements.txt
```

### Data preparation
This project uses PostgreSQL (v 14+), we also need [pgvector](https://github.com/pgvector/pgvector) extension to enable storage and retrieval of vector.

First install Postgres, create a database called `alardict`, user called `alaradmin`.

Run the `schema.sql` script to install schema:
```bash
psql -U alaradmin -d alardict < schema.sql
```

Now we can populate the required tables with data:
```bash
# TODO mention steps to download data.
```

## Running the server:
After the installation, you are all set to run the flask application:
```bash
DATABASE_URL=postgresql://alaradmin:<YOUR_PASSWORD>@localhost:5432/alardict python flaskapp.py
```

Head over to http://127.0.0.1:5000/ and the website should display homepage.


## Features
1. Direct English word search:
You can search for a single word such as `intelligence` and it gives you relevant, top 10 Kannada words.
<img width="819" alt="results-for-intelligence" src="https://github.com/sumitj39/elar.ink/assets/13235252/ecd82728-0f31-4631-a9cb-1178615c7624">

2. Related word search:
In case you do not remember the right word you can type in a sentence and it tries to give you relevant answer:
<img width="816" alt="results-for-loss-of-hearing" src="https://github.com/sumitj39/elar.ink/assets/13235252/021d1f17-8494-4789-bafe-b7f2cc0f62ef">


## Further enhancements
- [ ] Add support for ranking search results such that commonly used words appear at the top
- [ ] Support for pagination
