Building on Linux 
=================
(WIP)
Preliminary Requirements
--------------------------

The following are *necessary* to compile CMOD and LASS:

- A C++11-supporting compiler (g++, clang),
- A C compiler (gcc ...),
- premake4 >= 4.3,
- libsndfile >= 1.0,
- libxerces-c >= 3.2, and
- muparser >= 2.X (developers: this should be updated!)

To compile with LASSIE, the following couple are *necessary* inclusions:

- GTK+ 2.4 < 3.24 (developers: also should be updated!) and
- GTKmm-2.4 >= 1.5.

Recommended
-----------

LASSIE will want to open up a terminal, either gnome-terminal or xterm, to provide an interactive view of CMOD running your program. One of these is likely installed already if you're using a desktop environment. Otherwise, install one or the other, or both. *Note that LASSIE is an X application.*

Installing requirements and recommendations on:
-----------------------------------------------

### Debian-likes
You should first choose a compiler, say:

    sudo apt install g++ gcc

Then install the following:

    sudo apt install build-essential premake4 libsndfile1 libsndfile1-dev libxerces-c3.2 libxerces-c-dev libmuparser2v5 libmuparser-dev xterm

<!-- TODO: RHEL, maybe -->

Installing DISSCO
-----------------
Just `git clone` this repo; explicitly:

    git clone https://github.com/tipei/DISSCO-2.1.0.git

Building
--------
In the project's root directory (by default: `path/to/DISSCO-X.X.X`), run the following:

    premake4 && make

To generate the release build, do `make config=release`.