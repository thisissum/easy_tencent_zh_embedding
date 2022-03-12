# -*- coding: utf-8 -*-
# @Author : Minghong Sun
# @Time   : 2022/3/12 17:12
import os
import pickle
import path
from typing import Any

import tqdm
import numpy as np
import cachetools

"""
对腾讯AILab开源的词向量进行封装，使用缓存 + 文件索引，无需加载全量数据即可使用所有词向量

Usage:
>>> from tencent_embedding import TencentEmbedding
>>> embedding_path = "path of tencent embedding txt"
>>> embedding = TencentEmbedding.from_raw(path.Path(embedding_path))
>>> embedding = TencentEmbedding.from_built(path.Path(embedding_path))
>>> embedding = TencentEmbedding(path.Path(embedding_path))
>>> embedding["你好"]
...     ...
"""


class TencentEmbedding(object):

    @classmethod
    def from_raw(
        cls,
        embedding_path: os.PathLike,
        lru_cache_size: int = 100000,
        index_dir: os.PathLike = "./emb_index",
        preload_vocab_path: os.PathLike = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        cls.build_index(embedding_path, index_dir)
        return cls(embedding_path, lru_cache_size, index_dir, preload_vocab_path, encoding)

    @classmethod
    def from_built(
        cls,
        embedding_path: os.PathLike,
        lru_cache_size: int = 100000,
        index_dir: os.PathLike = "./emb_index",
        preload_vocab_path: os.PathLike = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        return cls(embedding_path, lru_cache_size, index_dir, preload_vocab_path, encoding)

    @classmethod
    def build_index(
        cls,
        embedding_path: os.PathLike,
        index_dir: os.PathLike = "./emb_index",
        encoding: str = "utf-8"
    ):
        if not os.path.exists(embedding_path):
            print("embedding_path: {} not exist".format(embedding_path))
            return None

        if not os.path.isdir(index_dir):
            os.mkdir(index_dir)

        print("Building index for: {}".format(embedding_path))
        # Dict[key: str, (key_id: int, start: int, offset: int)]
        key2index = {}
        index_file_name = "index.pkl"
        loop_cnt = 0

        with open(embedding_path, "rb") as f:
            start, end = 0, 0
            bar = tqdm.tqdm()
            while True:
                f.seek(start)
                line = f.readline()
                if not line:
                    break
                end = f.tell()
                offset = end - start
                index = (loop_cnt, start, offset)
                start = end
                if loop_cnt == 0:
                    vocab_size, emb_size = map(int, line.decode(encoding).split(" "))
                    bar.reset(vocab_size)
                try:
                    if loop_cnt > 0:
                        key2emb = line.split(b" ")
                        key = key2emb[0].decode(encoding)
                        key2index[key] = index
                except UnicodeDecodeError or MemoryError:
                    print("Error occurred when reading line: ")
                    print(line)

                loop_cnt += 1
                bar.update()

        # dump index with pickle
        print("Finished building index for: {}".format(embedding_path))
        index_output_path = os.path.join(index_dir, index_file_name)
        with open(index_output_path, "wb") as f:
            pickle.dump(key2index, f)
        print("Dumped index to path: {}".format(index_output_path))

    def __init__(
        self,
        embedding_path: os.PathLike,
        lru_cache_size: int = 100000,
        index_dir: os.PathLike = "./emb_index",
        preload_vocab_path: os.PathLike = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        self.embedding_path = embedding_path
        self.lru_cache_size = lru_cache_size
        self.index_dir = index_dir
        self.preload_vocab_path = preload_vocab_path
        self.encoding = encoding

        self._fp = open(self.embedding_path, "rb")
        self._lru_cache = cachetools.LRUCache(lru_cache_size)
        self._preload_cache = {}
        self._key2index = {}

        self._build()

    def _build(self):
        print("Initializing tencent embedding...", end="")
        index_path = os.path.join(self.index_dir, "index.pkl")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self._key2index.update(pickle.load(f))
                print("Done!", flush=True)

        if self.preload_vocab_path is not None and os.path.exists(self.preload_vocab_path):
            print("Loading preload vocab from path: {}...".format(self.preload_vocab_path), end="")
            with open(self.preload_vocab_path, "r", encoding=self.encoding) as f:
                preload_vocab = map(lambda x: x.strip().replace("\n", ""), f.readlines())
                for word in preload_vocab:
                    if word in self._key2index:
                        self._preload_cache[word] = self.get_from_disk(word)
            print("Done!", flush=True)

    def get_from_disk(self, key) -> np.ndarray:
        key_id, start, offset = self._key2index[key]
        self._fp.seek(start)
        line = self._fp.read(offset)
        output = np.asarray(line.decode(self.encoding).split(" ")[1:]).astype(float)
        return output

    def get_and_cache_from_disk(self, key) -> np.ndarray:
        output = self.get_from_disk(key)
        self._lru_cache[key] = output
        return output

    def get(self, key, default: Any = None):
        if key not in self._key2index:
            if default is None:
                raise KeyError("key: {} not found".format(key))
            return default
        else:
            output = self._preload_cache.get(
                key,
                self._lru_cache.get(
                    key,
                    self.get_and_cache_from_disk(key)
                )
            )
            return output

    def __getitem__(self, key: str) -> np.ndarray:
        return self.get(key)

    @property
    def keys(self):
        return self._key2index.keys()

    @property
    def cached_keys(self):
        cached_keys = set()
        cached_keys.update(self._preload_cache.keys())
        cached_keys.update(self._lru_cache.keys())
        return cached_keys

    def __del__(self):
        self._fp.close()
