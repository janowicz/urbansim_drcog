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

# Modify to RM = del for win
RM = rm -f
#RM = del

all: all_variables_edit_ui.py all_variables_select_ui.py all_variables_new_ui.py

clean:
	$(RM) all_variables_edit_ui.py all_variables_select_ui.py all_variables_new_ui.py
	$(RM) *.pyc
	$(RM) *~

all_variables_edit_ui.py: all_variables_edit.ui
	pyuic4 -o all_variables_edit_ui.py all_variables_edit.ui

all_variables_select_ui.py: all_variables_select.ui
	pyuic4 -o all_variables_select_ui.py all_variables_select.ui

all_variables_new_ui.py: all_variables_new.ui
	pyuic4 -o all_variables_new_ui.py all_variables_new.ui

