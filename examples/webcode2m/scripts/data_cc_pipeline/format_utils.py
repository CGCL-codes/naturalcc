from tqdm import tqdm
import re
import os
import glob
import time
import requests
from io import BytesIO
from PIL import Image,ImageDraw
from bs4 import BeautifulSoup,Comment
from pyquery import PyQuery as pq
import cssutils
import logging
import asyncio
import json
import shutil
from tools.log import logger
cssutils.log.setLevel(logging.CRITICAL)

def getCss(url):
    try:
        # 发送HTTP请求并获取响应
        logger.debug(f"Get css {url}.")        
        response = requests.get(url,timeout=1)
        # 检查链接状态码
        if response.status_code == 200:
            # 链接有效，下载CSS文本
            css_text = response.text
            return css_text
        else:
            # 链接失效
            logger.debug(f"CSS download error: {response.status_code}.")
            return False
    except Exception as e:
        logger.debug(f"CSS download error: {e}")
        return False
    
def splitHtmlCss(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    URI = soup(text=lambda text: isinstance(text, Comment))[0].strip()

    for link_tag in soup.find_all('link'):
        if 'href' in link_tag:
            href = link_tag['href']
            if href.endswith('.css'):
                if not href.startswith('http'):
                    href = URI+href
                css = getCss(href)
                if css:
                    new_style_tag = soup.new_tag('style')
                    new_style_tag.string = css
                    soup.body.append(new_style_tag)
        link_tag.decompose()
        
    style_contents = [style_tag.string for style_tag in soup.find_all('style') if style_tag.string]
    css = '\n'.join(style_contents)
    for style_tag in soup.find_all('style'):
        style_tag.decompose()
    
    html = re.sub(r'\n\s*\n', '\n', str(soup), flags=re.MULTILINE)
    html = '<!DOCTYPE html>\n'+str(html).strip()
    
    return html,css,URI

def formatHtml(html,URI):
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['meta', 'script']):
        tag.decompose()

    for element in soup(text=lambda text: isinstance(text, Comment)):
        element.extract()

    for tag in soup.find_all():
        preserved_attributes = ['class', 'id', 'width', 'height', 'style', 'src']
        for attribute in list(tag.attrs.keys()):
            if attribute not in preserved_attributes:
                del tag[attribute]

    for a_tag in soup.find_all('a'):
        del a_tag['href']
    
    """
    for img_tag in soup.find_all('img'):
        img_tag['src'] = img_placeholder_path
    """
            
    return str(soup).strip()

def formatCss(css, html):
    pattern = r'@import\s+url\([^\)]+\)'
    css =  re.sub(pattern, '', css)
    css_sheet = cssutils.parseString(css)
    doc = pq(html)
    rules = css_sheet.cssRules
    for rule in rules[:]: 
        if not isinstance(rule, cssutils.css.CSSStyleRule):
            rules.remove(rule)
        elif ':' in rule.selectorText:
            rules.remove(rule)
        elif doc and not doc(rule.selectorText):
            rules.remove(rule)
        elif rule.selectorText == 'html' and 'display' in rule.style:
            rules.remove(rule)
        else:
            # Remove properties starting with '-' or '--' or containing 'url('
            for prop in list(rule.style):
                if 'url(' in prop.value or prop.name.startswith('-'):
                    rule.style.removeProperty(prop.name)
                    
    return css_sheet.cssText.decode('utf-8').strip()

def mergeHtmlCss(html,css):
    soup = BeautifulSoup(html, 'html.parser')

    new_style_tag = soup.new_tag('style')
    new_style_tag.string = css.replace('@charset "UTF-8";','')
    soup.body.append(new_style_tag)
            
    return str(soup)

def move_style_inline(html_content):
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取style标签中的样式
    style_tag = soup.find('style')
    if style_tag:
        css_content = str(style_tag.string)
        styles = cssutils.parseString(css_content)
        
        for tag in soup.find_all():
            if 'style' in tag.attrs:
                tag['style'] = tag.attrs['style']
            else:
                tag['style'] = '' 

        for rule in styles:
            if rule.type == rule.STYLE_RULE:
                selectors = rule.selectorText.split(',')
                for selector in selectors:
                    selector = selector.strip()
                    elements = soup.select(selector)
                    for element in elements:
                        element['style'] +=  '; '+rule.style.cssText if element['style'] else rule.style.cssText
                        
        for tag in soup.find_all():
            if tag['style']:
                modified_css = re.sub(r'/\*.*?\*/', '', tag['style'], flags=re.DOTALL)
                modified_css = re.sub(r'\s+', ' ', modified_css, flags=re.MULTILINE)
                tag['style'] = modified_css
            else:
                del tag['style']
            del tag['class']
            del tag['id']

        style_tag.decompose()

    html = re.sub(r'\n\s*\n', '\n', str(soup), flags=re.MULTILINE)
    return html



    









