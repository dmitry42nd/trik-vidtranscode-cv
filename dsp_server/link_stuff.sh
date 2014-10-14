#!/bin/sh

/opt/trik-dsp/cgt_c6000_7.4.2/bin/lnk6x -w -q -u _c_int00 -fs package/cfg/bin/ -l link.cmd -q -o bin/dsp_server.xe674 package/cfg/bin/dsp_server_pe674.oe674  package/cfg/bin/dsp_server/main.oe674  package/cfg/bin/dsp_server_pe674.xdl --abi=eabi -c -m package/cfg//bin/dsp_server.xe674.map -l /opt/trik-dsp/cgt_c6000_7.4.2/lib/rts6740_elf.lib -l /opt/trik-dsp/c64plus-imglib_2_02_00_00/lib/target/imglib2_elf.lib -l /opt/trik-dsp/vlib_c674x_3_1_0_9/packages/ti/vlib/lib/vlib.lib
