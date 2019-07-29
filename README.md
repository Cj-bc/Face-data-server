README.md日本語: [JA_README.md](JA_README.md)

Japanese document is more accurate(as I'm Japanese) so I recommend you to read them first if you can.

---

# A face data server (temp name)

[![Build Status](https://travis-ci.com/Cj-bc/Face-data-server.svg?branch=develop)](https://travis-ci.com/Cj-bc/Face-data-server)

[![tipmona](https://img.shields.io/badge/tipme-%40tipmona-orange.svg)](https://twitter.com/share?text=%40tipmona%20tip%20%40Cj-bc%2039)  [![monya/mona](https://img.shields.io/badge/tipme-%40monya/mona-orange.svg)](https://monya-wallet.github.io/a/?address=MBdCkYyfTsCxtm1wZ1XyKWNLFLYj8zMK3V&scheme=monacoin)  [![tipkotone](https://img.shields.io/badge/tipme-%40tipkotone-orange.svg)](https://twitter.com/share?text=%40tipkotone%20tip%20%40Cj-bc%20)


This is piece of [Yozakura Project](https://github.com/Cj-bc/yozakura-project)

# About

This project aims to make a server that streams face data.
*face data* is series of data which points the position of each face parts


# Installation

## Make

This will install `face-data-server` to `/usr/local/bin` by default.
You can change `/usr/local` by providing `PREFIX` argument.

```bash
$ make install
```

## Manually

All what to do are:

```bash
$ git clone https://github.com/Cj-bc/Face-data-server
$ cd Face-dataserver
$ pipenv install
```

# usage

1st one is preferred.

```bash
$ ./face-data-server
# or
$ make run
# or
$ pipenv run python main.py
```

# Front end for this server

Currently, I want to make 3 official front end for different UI
