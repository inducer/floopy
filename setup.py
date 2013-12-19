#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    from setuptools import setup

    setup(name="floopy",
          version="2013.1",
          description="An AJAXy notes manager",
          #long_description=open("README.rst", "rt").read(),
          author=u"Andreas Kloeckner",
          author_email="inform@tiker.net",
          license="MIT",
          zip_safe=False,

          scripts=["bin/floopy"],
          packages=[
                  "floopy",
                  ],
          )

if __name__ == "__main__":
    main()
