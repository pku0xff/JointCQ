# The following code was adapted from https://github.com/hwchase17/langchain/blob/master/langchain/utilities/google_serper.py

"""Util that calls Google Search using the Serper.dev API."""
import argparse
import pdb

import requests
import asyncio
import aiohttp
import yaml
import os
import json

import jsonlines
from tqdm import tqdm, trange


class GoogleSerperAPIWrapper():
    """Wrapper around the Serper.dev Google Search API.
    You can create a free API key at https://serper.dev.
    To use, you should have the environment variable ``SERPER_API_KEY``
    set with your API key, or pass `serper_api_key` as a named parameter
    to the constructor.
    Example:
        .. code-block:: python
            from langchain import GoogleSerperAPIWrapper
            google_serper = GoogleSerperAPIWrapper()
    """

    def __init__(self, snippet_cnt=10, country='us', language='en') -> None:
        self.k = snippet_cnt
        self.gl = country
        self.hl = language
        # country='cn', hl='zh-cn'
        self.serper_api_key = os.environ.get("SERPER_API_KEY", None)
        assert self.serper_api_key is not None, "Please set the SERPER_API_KEY environment variable."
        assert self.serper_api_key != '', "Please set the SERPER_API_KEY environment variable."

    async def _google_serper_search_results(self, session, search_term: str, gl: str, hl: str) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        params = {"q": search_term, "gl": gl, "hl": hl}
        async with session.post(
                "https://google.serper.dev/search", headers=headers, params=params, raise_for_status=True
        ) as response:
            return await response.json()

    def _parse_results(self, results):
        snippets = []

        if results.get("answerBox"):
            answer_box = results.get("answerBox", {})
            if answer_box.get("answer"):
                element = {"content": answer_box.get("answer"), "source": "None"}
                return [element]
            elif answer_box.get("snippet"):
                element = {"content": answer_box.get("snippet").replace("\n", " "), "source": "None"}
                return [element]
            elif answer_box.get("snippetHighlighted"):
                element = {"content": answer_box.get("snippetHighlighted"), "source": "None"}
                return [element]

        if results.get("knowledgeGraph"):
            kg = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                element = {"content": f"{title}: {entity_type}", "source": "None"}
                snippets.append(element)
            description = kg.get("description")
            if description:
                element = {"content": description, "source": "None"}
                snippets.append(element)
            for attribute, value in kg.get("attributes", {}).items():
                element = {"content": f"{attribute}: {value}", "source": "None"}
                snippets.append(element)

        if results.get("organic"):
            for result in results["organic"][: self.k]:
                if "snippet" in result:
                    element = {"content": result["snippet"], "source": result["link"]}
                    snippets.append(element)
                for attribute, value in result.get("attributes", {}).items():
                    element = {"content": f"{attribute}: {value}", "source": result["link"]}
                    snippets.append(element)

        if len(snippets) == 0:
            element = {"content": "No good Google Search Result was found", "source": "None"}
            return [element]

        # keep only the first k snippets
        # snippets = snippets[:int(self.k / 2)]

        return snippets

    async def parallel_searches(self, search_queries, gl, hl):
        async with aiohttp.ClientSession() as session:
            tasks = [self._google_serper_search_results(session, query, gl, hl) for query in search_queries]
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            return search_results

    async def run(self, queries):
        """Run query through GoogleSearch and parse result."""
        flattened_queries = []

        for sublist in queries:
            if sublist is None:
                sublist = ['None', 'None']
            for item in sublist:
                flattened_queries.append(item)

        results = await self.parallel_searches(flattened_queries, gl=self.gl, hl=self.hl)
        snippets_list = []
        for i in range(len(results)):
            snippets_list.append(self._parse_results(results[i]))
        snippets_split = [snippets_list[i] + snippets_list[i + 1] for i in range(0, len(snippets_list), 2)]
        return snippets_split

    async def _run_grouped(self, queries):
        """Run grouped queries through GoogleSearch and return grouped results."""
        group_results = []
        async with aiohttp.ClientSession() as session:
            for query in queries:
                try:
                    raw_result = await self._google_serper_search_results(session, query, self.gl, self.hl)
                    parsed_result = self._parse_results(raw_result)
                    group_results.append({"query": query, "results": parsed_result})
                except Exception as e:
                    group_results.append({"query": query, "results": [{"content": str(e), "source": "Error"}]})
        return group_results

    def google_serper_search_results(self, query, gl, hl):
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        payload = json.dumps({"q": query, "gl": gl, "hl": hl})
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()

    def run_single(self, query):
        raw_result = self.google_serper_search_results(query, self.gl, self.hl)
        parsed_result = self._parse_results(raw_result)
        return parsed_result

    def run_grouped(self, queries):
        return asyncio.run(self._run_grouped(queries))


def load_queries(filename):
    with open(filename) as f:
        data = json.load(f)
    queries = []
    q_cnt = 0
    for item in data:
        for extraction in item['claim_extraction']:
            q_cnt += len(extraction['queries'])
            for q in extraction['queries']:
                queries.append(q)

    print('N queries: {}'.format(q_cnt))
    print('N data: {}'.format(len(data)))

    en_queries = []
    zh_queries = []
    for q in queries:
        if detect_language(q) == 'zh':
            zh_queries.append(q)
        else:
            en_queries.append(q)

    return en_queries, zh_queries


def postprocess(in_file, q2s, out_file):
    with open(in_file) as f:
        data = json.load(f)
    for i in range(len(data)):
        for j in range(len(data[i]['claim_extraction'])):
            curr_search_results = []
            for q in data[i]['claim_extraction'][j]['queries']:
                curr_search_results.append(q2s[q])
            data[i]['claim_extraction'][j]['search_results'] = curr_search_results
    with open(out_file, 'w') as f:
        json.dump(data, f, indent=2)


def detect_language(text):
    chinese_count = 0
    english_count = 0

    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
            chinese_count += 1
        elif char.isalpha():  # 英文字符（包括大小写）
            english_count += 1

    if chinese_count > english_count:
        return "zh"
    else:
        return "en"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--cache_file', type=str, default=None)
    parser.add_argument('--group_size', type=int, default=8)

    args = parser.parse_args()
    output_dir = args.output_dir
    dataset = args.dataset
    cache_file = args.cache_file
    group_size = args.group_size

    ceqg_file = f'ceqg_{dataset}.json'
    file_path = os.path.join(output_dir, ceqg_file)

    en_queries, zh_queries = load_queries(file_path)

    if cache_file is not None:
        if os.path.exists(cache_file):
            with jsonlines.open(cache_file, 'r') as f:
                cache_data = list(f)
        else:
            cache_data = []
    else:
        cache_data = []

    en_searcher = GoogleSerperAPIWrapper(country='us', language='en')
    zh_searcher = GoogleSerperAPIWrapper(country='cn', language='zh-cn')

    query2results = {i['query']: i['results'] for i in cache_data}
    en_queries = [q for q in en_queries if q not in query2results.keys()]
    zh_queries = [q for q in zh_queries if q not in query2results.keys()]
    print(len(en_queries), 'en queries to search')
    print(len(zh_queries), 'zh queries to search')
    with jsonlines.open(cache_file, 'a') as writer:
        print('Processing English queries ...')
        for i in trange(0, len(en_queries), group_size):
            query_group = [q for q in en_queries[i: i + group_size]]
            results_group = en_searcher.run_grouped(query_group)
            for item in results_group:
                writer.write(item)
                query2results[item['query']] = item['results']
        print('Processing Chinese queries ...')
        for i in trange(0, len(zh_queries), group_size):
            query_group = [q for q in zh_queries[i: i + group_size]]
            results_group = zh_searcher.run_grouped(query_group)
            for item in results_group:
                writer.write(item)
                query2results[item['query']] = item['results']

    postprocess(
        file_path,
        query2results,
        os.path.join(output_dir, f'search_{dataset}.json'),
    )
