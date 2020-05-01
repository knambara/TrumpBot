from ..items import TrumpItem

import scrapy
import json


class RemarkSpider(scrapy.Spider):

    name = "remark"
    start_urls = [
        'https://factba.se/json/json-transcript.php?q=&p=1&f=&in=&dt='
    ]
    base_url = 'https://factba.se/transcript/'
    json_url = 'https://factba.se/json/json-transcript.php?q=&p=%s&f=&in=&dt='
    page_num = 1

    def parse(self, response):

        # 145 pages of json available
        if self.page_num == 146:
            return
        response = json.loads(response.body_as_unicode())
        data = response["data"]

        for i, obj in enumerate(data):
            slug = obj["slug"]
            id = str(self.page_num * i)
            remark_page = self.base_url + slug
            request = scrapy.Request(remark_page, callback=self.scrape)
            request.meta['id'] = id
            yield request

        self.page_num += 1
        next_json = self.json_url % (str(self.page_num))
        yield scrapy.Request(next_json, callback=self.parse)

    def scrape(self, response):

        media = response.css('div.media')
        id = response.meta.get('id')

        for media_item in media:
            transcript = TrumpItem()
            transcript['text'] = media_item.css(
                'div.transcript-text-block a::text').get()
            transcript['speaker'] = media_item.css(
                'div.speaker-label::text').get()
            transcript['sentiment'] = media_item.css(
                'div.sentiment-block div::text').get()
            transcript['id'] = id

            yield transcript
