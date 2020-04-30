# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class TrumpItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    speaker = scrapy.Field()
    text = scrapy.Field()
    sentiment = scrapy.Field()
    id = scrapy.Field()
