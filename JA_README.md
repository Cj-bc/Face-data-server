English: [README.md](README.md)

---

# A face data server

[![Build Status](https://travis-ci.com/Cj-bc/Face-data-server.svg?branch=master)](https://travis-ci.com/Cj-bc/Face-data-server)

[![tipmona](https://img.shields.io/badge/tipme-%40tipmona-orange.svg)](https://twitter.com/share?text=%40tipmona%20tip%20%40Cj-bc%2039)  [![monya/mona](https://img.shields.io/badge/tipme-%40monya/mona-orange.svg)](https://monya-wallet.github.io/a/?address=MBdCkYyfTsCxtm1wZ1XyKWNLFLYj8zMK3V&scheme=monacoin)  [![tipkotone](https://img.shields.io/badge/tipme-%40tipkotone-orange.svg)](https://twitter.com/share?text=%40tipkotone%20tip%20%40Cj-bc%20)


これは[Yozakura Project](https://github.com/Cj-bc/yozakura-project)の一環です

# 概要

このプロジェクトの目的は、フェイスデータをストリーミング配信するサーバーの作成です。
*フェイスデータ*: 顔の各パーツの位置情報の集まり

詳しいデータについては[Cj-bc/FDS-protos](https://github.com/Cj-bc/FDS-protos)

# Installation

## Make

デフォルトで`/usr/local/bin`以下にインストールします。
`PREFIX`変数を与えることで`/usr/local`部を変更できます。

```bash
$ make install
```

## 手動

具体的に行わなければいけない工程は以下の通りです。

```bash
$ git clone https://github.com/Cj-bc/Face-data-server
$ cd Face-dataserver
$ pipenv install
```

# 使い方

最初の方法をお勧めします。

```bash
$ ./face-data-server
# もしくは
$ make run
# もしくは
$ pipenv run python main.py
```

# フロントエンド

- [Cj-bc/faclig](https://github.com/Cj-bc/faclig) -- ASCII Artモデルを動かすフロントエンドです

...and more
