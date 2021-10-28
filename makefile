
all: __main__.py models/deepsquare-k16-3.pb
	echo '#!/usr/bin/env python' > psb
	zip psb.zip __main__.py models/deepsquare-k16-3.pb
	cat psb.zip >> psb
	rm psb.zip
	chmod +x psb

clean: psb
	rm -f psb

install: psb
	mkdir -p /usr/local/bin/
	cp psb /usr/local/bin/
