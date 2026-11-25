ENTRY(custom_job)
/* The firmware vector table the entry stubs forward to (sdk/entry.S). */
firmware_vectors = 0x87000000;
SECTIONS {
 . = 0xf0a92000;
 .payload : { *(.text.entry) *(.text*) *(.rodata*) *(.data*) *(.sdata*) }
 .bss : { *(.bss*) *(.sbss*) *(COMMON) }
 ASSERT(SIZEOF(.payload) <= 0x1000, "DSP payload exceeds 4 KiB code reservation")
 ASSERT(SIZEOF(.bss) == 0, "DSP kernels must not depend on BSS initialization")
 /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
