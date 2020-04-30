# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


class TrumpPipeline:

    def process_item(self, item, spider):

        if item.get('speaker') == "Note":
            return
        if item.get('text') == None:
            return

        id = item.get('id')
        file_name = 'interviews/trump_interview_%s.txt' % (id)

        with open(file_name, 'a+') as f:
            speaker = item.get('speaker').strip()
            sentiment = item.get('sentiment').strip()
            if sentiment == "" or sentiment == "Unknown":
                sentiment = "Neutral"
            text = item.get('text').strip()

            body = "<%s> <%s> %s" % (speaker, sentiment, text)
            f.write('%s\n' % (body))
