#
# UrbanSim software. Copyright (C) 1998-2007 University of Washington
# 
# You can redistribute this program and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation
# (http://www.gnu.org/copyleft/gpl.html).
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the file LICENSE.html for copyright
# and licensing information, and the file ACKNOWLEDGMENTS.html for funding and
# other acknowledgments.
# 

# Modify for win
RM = rm -f
#RM = del

all:
	$(MAKE) -C main
	$(MAKE) -C util
	$(MAKE) -C config
	$(MAKE) -C run
	$(MAKE) -C map
	$(MAKE) -C settings

clean:
	$(RM) *.pyc
	$(RM) *~
	$(MAKE) -C main clean
	$(MAKE) -C util clean
	$(MAKE) -C config clean
	$(MAKE) -C run clean
	$(MAKE) -C map clean
	$(MAKE) -C settings clean
