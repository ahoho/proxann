# Annotation Server for Qualtrics

We use Qualtrics surveys for item annotation, but Qualtrics is not designed for this purpose.

We populate the survey data using a json containing a random sample of items that the respondent will annotate. This json is produced by a separate Flask server (run on [AWS elastic beanstalk](https://aws.amazon.com/elasticbeanstalk/)). 

`setup_annotation.py` here creates a zipped folder containing the server code and annotation data that can then be uploaded and served from AWS (although any provider would be fine---it's very simple and lightweight). Here is what was run to create the server:

```bash
python setup_annotation.py \
    ../../data/json_out/config_wiki_part1.json \
    ../../data/json_out/config_wiki_part2.json \
    ../../data/json_out/config_bills_part1.json \
    ../../data/json_out/config_bills_part2.json
```

We will provide the Qualtrics survey template as soon as we can---some temporary technical issues have prevented us from accessing it (please create an issue if we have neglected to do so!)