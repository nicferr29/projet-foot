if {[package vsatisfies [package provide Tcl] 9.0-]} { 
package ifneeded sqlite3 3.53.0 [list load [file join $dir tcl9sqlite3530.dll] [string totitle sqlite3]] 
} else { 
package ifneeded sqlite3 3.53.0 [list load [file join $dir sqlite3530t.dll] [string totitle sqlite3]] 
} 
