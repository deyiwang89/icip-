- 主要内容源自[教程：WRF 3.9.1.1 在Ubuntu16.04 LTS 下的安装包括Chem kpp](http://bbs.06climate.com/forum.php?mod=viewthread&tid=57144&highlight=UBUNTU%2BWRF), 并进行修改, 经测试, 所有命令均有效.
- 本配置步骤涵盖WRF及无KPP的WRF-Chem编译, 不包括WPS及WRF-DA
##  WRFV3.9.1.1配置步骤
- 默认配置: 系统：Ubuntu 16.04 LTS(cat /etc/issue 查看)
WRF：V3.9.1.1 （WRFV3.9.1.1.TAR.gz；WRFV3-Chem-3.9.1.TAR.gz；WPSV3.9.1.TAR.gz）
NETCDF (NetCDF-C V4.4.1; NetCDF-FORTRAN V4.4.4)
HDF5: V1.8.18; NCL: V6.4.0
库函数：zlib: V1.2.10;Jasper: V1.900.1 (要是版本过高会不生成ungrid.exe, 也有解决办法，后文提到) ; PNG:V1.6.26;Libjpeg.v9a
gcc 5.4.0  g++ 5.4.0  gfortran 5.4.0
byacc.1.9.tar.Z  wgrib2.tgz  
以上所有安装包均在`/home/icip1004/下载`文件夹内.
- 更新系统软件:`sudo apt-get upgrade`
### 安装编译工具:
- 打开`ubuntuSoftware`,  查找`新的软件包管理器(synaptic)`, 点击安装, 然后等待自行完成.
- 在`ubuntuSoftware`中查找`perl`, 使用命令`perl -v`, 得到如下返回:
```
This is perl 5, version 22, subversion 1 (v5.22.1) built for x86_64-linux-gnu-thread-multi
(with 58 registered patches, see perl -V for more detail)
```
- 检查awk, 命令:`which awk`, 返回`/usr/bin/awk`
- 安装`tcsh samba cpp m4 quota`
命令:`sudo apt-get install tcsh samba cpp m4 quota`
- 检查`tcsh`是否安装好:`tcsh --version`
```
tcsh 6.18.01 (Astron) 2012-02-14 (x86_64-unknown-linux) options wide,nls,dl,al,kan,rh,nd,color,filec
```
- 检查`samba`, 命令:`samba --version`
```
Version 4.3.11-Ubuntu
```
- cpp: `cpp --version`
```
cpp (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
```
- `m4 --version`
```
m4 (GNU M4) 1.4.17
```
### 安装WRF编译工具

- 1）首先判断gcc，g++，gfortran版本是否一致
```
gcc --version		
echo "5.4.0"
g++ --version
echo "5.4.0"	
gfortran --version		
sudo apt-get install gfortran
echo  "5.4.0"
```
- 2）重新连接一遍，不然会寻址不到.
**【重新链接gcc，g++，gfortran】**
1.建立备份文件夹：
`sudo -i` #进入管理员模式
`mkdir /usr/bin/gccbackup`
`mkdir /usr/bin/g++backup`
`mkdir /usr/bin/gfortranbackup`
2.先将原来的链接改名，终端输入：
`mv /usr/bin/gcc /usr/bin/gccbackup`
`mv /usr/bin/g++ /usr/bin/g++backup`
`mv /usr/bin/gfortran /usr/bin/gfortranbackup`
3.重新链接
`ln -s /usr/bin/gcc-5 /usr/bin/gcc`
`ln -s /usr/bin/gfortran-5 /usr/bin/gfortran`
`ln -s /usr/bin/g++-5 /usr/bin/g++`
### 安装NetCDF
NetCDF的安装需要有HDF5lib，而HDF5的安装需要先有zlib和curl.所以先要安装zlib和curl.本人将zlib，curl,  jpeg, png.  Jasper 一起安装了，并且建立了一个JASPER文件夹将这几个lib与include都放在一起，方便后期的环境设置.
打开`synaptic`，点击搜索，输入`libjpeg8`，然后`libjpeg8-dbg/libjpeg8-dev`选项，前面没有打钩的.双击打钩标记，标记后会变绿色，点击应用，然后安装.
同理输入`glibc`，然后就会看到结果有三个红点的`glibc`选项，也`apply`.`-doc -doc-reference -source`
同理输入`grib2`，然后就会看到结果有`libgrib2c-dev/libgrib2c0d`选项，也`apply`.
### 开始安装各个小东西
第一步解压，把下载好的都解压了，我这里是解压到/usr/local/src,安装完可以删除
```
tar -zxf zlib-1.2.10.tar.gz -C /usr/local/src
tar -zxf jpegsrc.v9a.tar.gz -C /usr/local/src
tar -zxf libpng-1.6.26.tar.gz -C /usr/local/src
unzip jasper-1.900.1.zip 
mv jasper-1.900.1 /usr/local/src
```
### 接下来就是各种安装了
1.       zlib
```
cd /usr/local/src/zlib-1.2.10/
./configure --prefix=/usr/local/zlib
make
make check
make install
```
返回:
```
rm -f /usr/local/zlib/lib/libz.a
cp libz.a /usr/local/zlib/lib
chmod 644 /usr/local/zlib/lib/libz.a
cp libz.so.1.2.10 /usr/local/zlib/lib
chmod 755 /usr/local/zlib/lib/libz.so.1.2.10
rm -f /usr/local/zlib/share/man/man3/zlib.3
cp zlib.3 /usr/local/zlib/share/man/man3
chmod 644 /usr/local/zlib/share/man/man3/zlib.3
rm -f /usr/local/zlib/lib/pkgconfig/zlib.pc
cp zlib.pc /usr/local/zlib/lib/pkgconfig
chmod 644 /usr/local/zlib/lib/pkgconfig/zlib.pc
rm -f /usr/local/zlib/include/zlib.h /usr/local/zlib/include/zconf.h
cp zlib.h zconf.h /usr/local/zlib/include
chmod 644 /usr/local/zlib/include/zlib.h /usr/local/zlib/include/zconf.h

```
2.       curl
`apt-get install curl`
3.       jpeg-9a
`cd /usr/local/src/jpeg-9a`
`./configure --prefix=/usr/local/libjpeg`
`make`
返回:`make[1]: Leaving directory '/usr/local/src/jpeg-9a'`
`make install`
返回
```
 /bin/mkdir -p '/usr/local/libjpeg/share/man/man1'
 /usr/bin/install -c -m 644 cjpeg.1 djpeg.1 jpegtran.1 rdjpgcom.1 wrjpgcom.1 '/usr/local/libjpeg/share/man/man1'
make[1]: Leaving directory '/usr/local/src/jpeg-9a'
```
4.       libpng
```
cd /usr/local/src/libpng-1.6.26
export LDFLAGS=-L/usr/local/zlib/lib
export CPPFLAGS=-I/usr/local/zlib/include
./configure --prefix=/usr/local/libpng
make
make install
```
返回:
```
+ cd /usr/local/libpng/lib/pkgconfig
+ rm -f libpng.pc
+ ln -s libpng16.pc libpng.pc
make[3]: Leaving directory '/usr/local/src/libpng-1.6.26'
make[2]: Leaving directory '/usr/local/src/libpng-1.6.26'
make[1]: Leaving directory '/usr/local/src/libpng-1.6.26'
```
5.       jasper-1.900.1
```
cd /usr/local/src/jasper-1.900.1
./configure --prefix=/usr/local/jasper 
make
make install
```
我安装的是1.9，因为高版本在后面可能出现不了ungrib.exe
高版本的Jasper需要修改jas_image.h文件, 本过程并未使用
```
cd /usr/local/jasper/include/jasper/
gedit jas_image.h（找到bool inmem_; 将前面的"//"去掉）(in this case there isn't a "//")
```
### 配置环境变量
```
mkdir /usr/local/JASPER
mkdir /usr/local/JASPER/lib
mkdir /usr/local/JASPER/include
cp -r /usr/local/zlib/lib/* /usr/local/JASPER/lib
cp -r /usr/local/libpng/lib/* /usr/local/JASPER/lib
cp -r /usr/local/jasper/lib/* /usr/local/JASPER/lib
cp -r /usr/local/zlib/include/* /usr/local/JASPER/include
cp -r /usr/local/libpng/include/* /usr/local/JASPER/include
cp -r /usr/local/jasper/include/* /usr/local/JASPER/include
```
`gedit ~/.bashrc`
在最后添加下面路径指向：
```
#for zlib
export ZLIB_HOME=/usr/local/zlib
export LD_LIBRARY_PATH=$ZLIB_HOME/lib:$LD_LIBRARY_PATH
#for libpng
export ZLIB_HOME=/usr/local/libpng
export LIBPNGLIB=/usr/local/libpng/lib
export LIBPNGINC=/usr/local/libpng/include
#set JASPER
export JASPER=/usr/local/JASPER
export JASPERLIB=/usr/local/JASPER/lib
export JASPERINC=/usr/local/JASPER/include
```
保存后退出
`source ~/.bashrc`
要是保存不了，就 chmod 修改一下权限
(JASPER的两个文件夹下分别有15个和7个文件)
### 安装配置hdf5
```
cd /home/icip1004/下载
gunzip hdf5-1.8.18.tar.gz 
tar -xf hdf5-1.8.18.tar -C /usr/local/src
cd /usr/local/src/hdf5-1.8.18
 ./configure --prefix=/usr/local/HDF5 --with-zlib=/usr/local/zlib
```
返回:
```
	    SUMMARY OF THE HDF5 CONFIGURATION
	    =================================

General Information:
-------------------
		   HDF5 Version: 1.8.18
		  Configured on: Sat Oct  5 22:52:55 CST 2019
		  Configured by: root@icip1004-All-Series
		 Configure mode: production
		    Host system: x86_64-unknown-linux-gnu
	      Uname information: Linux icip1004-All-Series 4.8.0-36-generic #36~16.04.1-Ubuntu SMP Sun Feb 5 09:39:57 UTC 2017 x86_64 x86_64 x86_64 GNU/Linux
		       Byte sex: little-endian
		      Libraries: static, shared
	     Installation point: /usr/local/HDF5

Compiling Options:
------------------
               Compilation Mode: production
                     C Compiler: /usr/bin/gcc
                         CFLAGS: 
                      H5_CFLAGS: -std=c99 -pedantic -Wall -Wextra -Wundef -Wshadow -Wpointer-arith -Wbad-function-cast -Wcast-qual -Wcast-align -Wwrite-strings -Wconversion -Waggregate-return -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wredundant-decls -Wnested-externs -Winline -Wfloat-equal -Wmissing-format-attribute -Wmissing-noreturn -Wpacked -Wdisabled-optimization -Wformat=2 -Wunreachable-code -Wendif-labels -Wdeclaration-after-statement -Wold-style-definition -Winvalid-pch -Wvariadic-macros -Winit-self -Wmissing-include-dirs -Wswitch-default -Wswitch-enum -Wunused-macros -Wunsafe-loop-optimizations -Wc++-compat -Wstrict-overflow -Wlogical-op -Wlarger-than=2048 -Wvla -Wsync-nand -Wframe-larger-than=16384 -Wpacked-bitfield-compat -Wstrict-overflow=5 -Wjump-misses-init -Wunsuffixed-float-constants -Wdouble-promotion -Wsuggest-attribute=const -Wtrampolines -Wstack-usage=8192 -Wvector-operation-performance -Wsuggest-attribute=pure -Wsuggest-attribute=noreturn -Wsuggest-attribute=format -Wdate-time -Wopenmp-simd -Warray-bounds=2 -Wc99-c11-compat -O3 -fstdarg-opt
                      AM_CFLAGS: 
                       CPPFLAGS: -I/usr/local/zlib/include
                    H5_CPPFLAGS: -D_GNU_SOURCE -D_POSIX_C_SOURCE=200112L   -DNDEBUG -UH5_DEBUG_API
                    AM_CPPFLAGS:  -I/usr/local/zlib/include
               Shared C Library: yes
               Static C Library: yes
  Statically Linked Executables: no
                        LDFLAGS: -L/usr/local/zlib/lib
                     H5_LDFLAGS: 
                     AM_LDFLAGS:  -L/usr/local/zlib/lib
 	 	Extra libraries: -lz -ldl -lm 
 		       Archiver: ar
 		 	 Ranlib: ranlib
 	      Debugged Packages: 
		    API Tracing: no

Languages:
----------
                        Fortran: no

                            C++: no

Features:
---------
                  Parallel HDF5: no
             High Level library: yes
                   Threadsafety: no
            Default API Mapping: v18
 With Deprecated Public Symbols: yes
         I/O filters (external): deflate(zlib)
                            MPE: no
                     Direct VFD: no
                        dmalloc: no
Clear file buffers before write: yes
           Using memory checker: no
         Function Stack Tracing: no
      Strict File Format Checks: no
   Optimization Instrumentation: no
```
接下来进行编译:
`make`
`make check`
`make install`
`make check-install` 
在bashrc中添加环境变量
`gedit ~/.bashrc`
```
# for hdf5
export CPPFLAGS=-I$PRO_PATH/usr/local/HDF5/include
export LDFLAGS=-L$PRO_PATH/usr/local/HDF5/lib
export LD_LIBRARY_PATH=$PRO_PATH/usr/local/HDF5/lib
```
使配置生效:
`source ~/.bashrc`

### 安装NetCDF-C
`cd /home/icip1004/下载`
```
tar -zxf netcdf-c-4.4.1.tar.gz -C /usr/local/src
cd /usr/local/src/netcdf-c-4.4.1/
export CPPFLAGS=-I/usr/local/HDF5/include
export LDFLAGS=-L/usr/local/HDF5/lib
export LD_LIBRARY_PATH=$/usr/local/HDF5/lib
```
`./configure --prefix=/usr/local/NETCDF --disable-netcdf-4`
开始编译
`make`
`make check`
`make install`
即可
### 安装NetCDF -fortran
`cd /home/icip1004/下载`
```
tar -xzf netcdf-fortran-4.4.4.tar.gz -C /usr/local/src
cd /usr/local/src/netcdf-fortran-4.4.4/
```
```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/NETCDF/lib
export rt CPPFLAGS=-I/usr/local/NETCDF/include
export LDFLAGS=-L/usr/local/NETCDF/lib
```
`./configure --prefix=/usr/local/NETCDF FC=gfortran`
开始编译
`make`
`make check`
`make install`
`gedit ~/.bashrc`
添加环境变量:
```
#for netcdf
export PATH=/usr/local/NETCDF/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/NETCDF/lib:$LD_LIBRARY_PATH
```
`source ~/.bashrc`
### 安装NCL
`mkdir /usr/local/ncarg`
`cd /home/icip1004/下载`
```
tar -zxf ncl_ncarg-6.4.0-Debian8.6_64bit_gnu492.tar.gz -C /usr/local/ncarg/
cd /usr/local/ncarg
```
`gedit ~/.bashrc`
在最后添加下列语句:
```
#for ncarg
export NCARG_ROOT=/usr/local/ncarg
export PATH=$NCARG_ROOT/bin:$PATH
export MANPATH=$NCARG_ROOT/man:$MANPATH
export DISPLAY=:0.0
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
#(这一句用于WRF输出大数据)
```
保存后退出
`source ~/.bashrc`
检验NCL是否成功安装，新终端输入：
`ncargversion`
如果出现如下情况:
```
icip1004@icip1004-All-Series:~$ ncargversion
程序"ncargversion”尚未安装. 您可以使用以下命令安装：
sudo apt install ncl-ncarg
```
需要同步在sudo -i下的~/.bashrc{$A}和/home/icip1004/.bashrc{$B}两个文件的内容, 并且同时使之生效, 命令:`source {$A}, source {$B}`
生效后的返回:
```
NCAR Graphics Software Version 6.4.0
Copyright (C) 1987-2017, University Corporation for Atmospheric Research

NCAR Graphics is a registered trademark of the University Corporation
for Atmospheric Research.

The use of this Software is governed by a License Agreement.
```
### 安装wgrib2
命令如下:
```
cd /home/icip1004/下载
cp wgrib2.tgz.v1.8.6 /usr/local
cd /usr/local
tar -zxf wgrib2.tgz.v1.8.6
cd grib2/
export CC=gcc
export FC=gfortran
make
```
### 安装WRF
命令如下:
```
mkdir /home/icip1004/model
gedit /home/icip1004/.bashrc
source /home/icip1004/.bashrc
sudo -i
cd /home/icip1004/下载
tar -xzf WRFV3.9.1.1.TAR.gz -C /home/icip1004/model
```
这时在/model出现了WRFV3的文件，进入WRFV3里面的arch文件夹，找到Config_new.pl文件，打开后，找到下面这句并做如下修改：
```
$I_really_want_to_output_grib2_from_WRF= "TRUE" ;
```
对应下面几行，找到并修改：
```
$I_really_want_to_output_grib2_from_WRF = "FALSE" ;
 $I_really_want_to_output_grib2_from_WRF = "TRUE" ;

 if ( $ENV{JASPERLIB} && $ENV{JASPERINC} && $I_really_want_to_output_grib2_from_WRF eq "TRUE" )
   {
   printf "Configuring to use jasper library to build Grib2 I/O...\n" ;
   printf("  \$JASPERLIB = %s\n",$ENV{JASPERLIB});
   printf("  \$JASPERINC = %s\n",$ENV{JASPERINC});
   $sw_jasperlib_path = "/usr/local/JASPER/lib";#$ENV{JASPERLIB}; 
   $sw_jasperinc_path = "/usr/local/JASPER/include";#$ENV{JASPERINC}; 
```
保存后退出.
```
cd /home/icip1004//model/WRFV3/
#need to copy every added lines from /home/icip1004/.bashrc to ~/.bashrc
./configure
```
先选择`32`后选择`1`

在\arch文件夹里面找到`configure_new.defaults`，打开，找到这一部分
```
#ARCH    Linux x86_64 ppc64le, gfortran compiler with gcc  #serial smpar dmpar dm+sm
FORMAT_FIXED    =       -ffixed-form
FORMAT_FREE     =       -ffree-form -ffree-line-length-none
-------------------
FORMAT_FIXED    =       -ffixed-form -cpp
FORMAT_FREE     =       -ffree-form -cpp -ffree-line-length-none
```
保存后退出
在WRFV3目录下找到configure.wrf文件并打开，做如下修改：（同上）
```
FORMAT_FIXED    =       -ffixed-form
FORMAT_FREE     =       -ffree-form -ffree-line-length-none
-------------------
FORMAT_FIXED    =       -ffixed-form -cpp
FORMAT_FREE     =       -ffree-form -cpp -ffree-line-length-none
```
保存后退出.
在终端输入：
```
./compile em_real >&checkwrf.log
ls -ls main/*.exe 
```
得到如下返回:
```
48584 -rwxr-xr-x 1 root 8079 49748400 10月  6 00:17 main/ndown.exe
48460 -rwxr-xr-x 1 root 8079 49621328 10月  6 00:17 main/real.exe
48080 -rwxr-xr-x 1 root 8079 49232560 10月  6 00:17 main/tc.exe
52056 -rwxr-xr-x 1 root 8079 53300984 10月  6 00:16 main/wrf.exe

```
### 安装KPP
KPP 需要 flex 和 yacc 的支持，这一步是安装 KPP 的关键，大多数 KPP 安装错误要么是flex 和 yacc 版本不对，要么是环境变量设置错误.
标准安装的顺序是先安装flex然后安装yacc，可是本人在安装flex时出现了无法执行yacc命令的错误，所以又颠倒顺序安装了一下. 如果发生错误, 将错误粘贴到网上，一般都会有解决办法的.
- Yacc:
```
cd /home/icip1004/下载
gzip -d byacc.1.9.tar.Z
mkdir /usr/local/yacc
tar -xf byacc.1.9.tar -C /usr/local/yacc
cd /usr/local/yacc/
make
```
- flex:
```
cd /home/icip1004/下载
mkdir /usr/local/flex
mv flex-2.5.37.tar.gz /usr/local/flex
cd /usr/local/flex
tar -xzf flex-2.5.37.tar.gz 
cd flex-2.5.37/
./configure --prefix=/usr/local/flex
make
make install
```
检验 flex 和 yacc的安装:
```
root@icip1004-All-Series:~# which flex
/usr/local/flex/bin/flex
root@icip1004-All-Series:~# which yacc
/usr/local/yacc/yacc
root@icip1004-All-Series:~# 
```
为Chem添加环境变量`gedit ~/.bashrc`
```
# for chem
export WRF_CHEM=1 
#（编译 WRF_Chem）
export WRF_KPP=1
#(安装 KPP，0 表示不安装 KPP，若不安装 KPP，下面的环境变量不需要设置)
export PATH=/usr/local/yacc:$PATH
export PATH=/usr/local/flex/bin:$PATH
export YACC='/usr/local/yacc/yacc -d'
export FLEX=/usr/local/flex/bin/flex
export FLEX_LIB_DIR=/usr/local/flex/lib
source ~/.bashrc
```
对 `/home/icip1004/.bashrc`进行同样的操作

### 安装 WRF-CHEM:
```
cd /home/icip1004/下载
tar -xzf WRFV3-Chem-3.9.1.TAR.gz -C /home/icip1004/model/WRFV3
cd /home/icip1004/model/WRFV3
./clean -a #(删除以前的 WRF 编译)
./configure 
./compile em_real >&checkwrf.log
```
得到返回:
```
*** buffer overflow detected ***: /home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp terminated
======= Backtrace: =========
/lib/x86_64-linux-gnu/libc.so.6(+0x777e5)[0x7fd4332907e5]
/lib/x86_64-linux-gnu/libc.so.6(__fortify_fail+0x5c)[0x7fd43333156c]
/lib/x86_64-linux-gnu/libc.so.6(+0x116570)[0x7fd43332f570]
/lib/x86_64-linux-gnu/libc.so.6(+0x1158c2)[0x7fd43332e8c2]
/home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp[0x404660]
/home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp[0x401502]
/home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp[0x40237e]
/home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp[0x4065b6]
/home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp[0x40836f]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf0)[0x7fd433239830]
/home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp[0x401189]
======= Memory map: ========
00400000-00429000 r-xp 00000000 08:07 2228906                            /home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp
00628000-00629000 r--p 00028000 08:07 2228906                            /home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp
00629000-0062f000 rw-p 00029000 08:07 2228906                            /home/icip1004/model/WRFV3/chem/KPP/kpp/kpp-2.1/bin/kpp
0062f000-006c9000 rw-p 00000000 00:00 0 
009ce000-010d2000 rw-p 00000000 00:00 0                                  [heap]
7fd433003000-7fd433019000 r-xp 00000000 08:07 1444365                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7fd433019000-7fd433218000 ---p 00016000 08:07 1444365                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7fd433218000-7fd433219000 rw-p 00015000 08:07 1444365                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7fd433219000-7fd4333d8000 r-xp 00000000 08:07 1444327                    /lib/x86_64-linux-gnu/libc-2.23.so
7fd4333d8000-7fd4335d8000 ---p 001bf000 08:07 1444327                    /lib/x86_64-linux-gnu/libc-2.23.so
7fd4335d8000-7fd4335dc000 r--p 001bf000 08:07 1444327                    /lib/x86_64-linux-gnu/libc-2.23.so
7fd4335dc000-7fd4335de000 rw-p 001c3000 08:07 1444327                    /lib/x86_64-linux-gnu/libc-2.23.so
7fd4335de000-7fd4335e2000 rw-p 00000000 00:00 0 
7fd4335e2000-7fd433608000 r-xp 00000000 08:07 1444299                    /lib/x86_64-linux-gnu/ld-2.23.so
7fd4337ec000-7fd4337ef000 rw-p 00000000 00:00 0 
7fd433804000-7fd433807000 rw-p 00000000 00:00 0 
7fd433807000-7fd433808000 r--p 00025000 08:07 1444299                    /lib/x86_64-linux-gnu/ld-2.23.so
7fd433808000-7fd433809000 rw-p 00026000 08:07 1444299                    /lib/x86_64-linux-gnu/ld-2.23.so
7fd433809000-7fd43380a000 rw-p 00000000 00:00 0 
7fff6398a000-7fff639ab000 rw-p 00000000 00:00 0                          [stack]
7fff639b1000-7fff639b3000 r--p 00000000 00:00 0                          [vvar]
7fff639b3000-7fff639b5000 r-xp 00000000 00:00 0                          [vdso]
ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0                  [vsyscall]
```