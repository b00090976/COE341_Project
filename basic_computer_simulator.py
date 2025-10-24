"""
Basic Computer Simulator
========================

This module implements a software‐only simulation of Mano’s Basic Computer
architecture as described in the COE 341 course assignment.  The goal of the
assignment is to expose students to the instruction cycle (fetch–decode–execute)
and to provide an environment where individual clock cycles and whole
instructions can be stepped through interactively.  The implementation here
strives to follow the specification provided in the project handout: it

* Loads program and data words from `program.txt` and `data.txt` files.
* Simulates the full instruction cycle for all memory‐reference and
  register‐reference instructions.  I/O instructions are not implemented.
* Provides cycle‐level, instruction‐level and program‐level execution controls
  (`next_cycle`, `fast_cycle`, `next_inst`, `fast_inst` and `run`).
* Allows inspection of registers and memory (`show <register>`, `show mem`)
  along with an aggregate `show all` command.
* Maintains a simple profiler tracking total clock cycles, instructions
  executed, and memory bandwidth (reads and writes).

The simulator is intentionally minimalist; it omits any graphical user
interface or assembler so that students can focus on the core behaviour of
the architecture without unnecessary complexity.  See the documentation in
`README.md` or the project handout for usage examples.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict, Iterable


class BasicComputerSimulator:
    """A software model of Mano's Basic Computer.

    The simulator maintains registers, a memory array, and internal state to
    progress through the fetch–decode–execute sequence one micro operation at
    a time.  Clients can control execution at the granularity of single
    cycles, complete instructions, or entire programs, and can inspect state
    after each command.
    """

    def __init__(self) -> None:
        # 4096 words of 16‑bit memory
        self.memory: List[int] = [0] * 4096

        # Registers (AC, DR, AR, IR, PC, TR) and single–bit flags (E, I)
        self.AC: int = 0  # Accumulator (16 bits)
        self.DR: int = 0  # Data Register (16 bits)
        self.AR: int = 0  # Address Register (12 bits)
        self.IR: int = 0  # Instruction Register (16 bits)
        self.PC: int = 0  # Program Counter (12 bits)
        self.TR: int = 0  # Temporary Register (16 bits) – unused in this simple model
        self.E: int = 0   # Carry flag (1 bit)
        self.I: int = 0   # Indirect flag (1 bit)

        # Internal micro–phase counter (0=T0, 1=T1, …)
        self.micro_phase: int = 0

        # Halting state – set when HLT register reference executes
        self.halted: bool = False

        # Profiling counters
        self.cycle_counter: int = 0       # Total clock cycles executed
        self.instruction_counter: int = 0 # Total instructions executed
        self.read_count: int = 0          # Memory reads
        self.write_count: int = 0         # Memory writes

        # Last executed instruction word (16‑bit) – updated each time an
        # instruction completes.  Useful for summarising instruction
        # execution in higher‑level commands.
        self.last_executed_instruction: Optional[int] = None

        # Address at which execution halts (HLT).  None while running
        self.halt_pc: Optional[int] = None

    # ------------------------------------------------------------------
    # Program and data loading
    #
    def load_program(self, program_file: str = "program.txt", data_file: Optional[str] = "data.txt") -> None:
        """Load program and data from text files.

        Each file should contain lines with a 12‑bit address (hex) and a 16‑bit
        word (hex) separated by whitespace, for example:

            100  2003
            101  7001

        Addresses may be prefixed by `0x`, and blank lines or comments
        beginning with `#` are ignored.  If a given address appears multiple
        times the last occurrence wins.  After loading, the program counter
        (PC) is initialised to the smallest address in the program file,
        defaulting to 0 if no lines are present.

        :param program_file: name of the program file to load
        :param data_file: optional name of the data file to load
        """
        # Helper to parse lines into (address, value)
        def parse_file(path: str) -> List[Tuple[int, int]]:
            entries: List[Tuple[int, int]] = []
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Remove comments and strip whitespace
                        line = line.split('#', 1)[0].strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        addr_str, value_str = parts[0], parts[1]
                        # Parse address and value as hexadecimal; allow optional 0x prefix
                        address = int(addr_str, 16)
                        value = int(value_str, 16)
                        if not (0 <= address < 4096):
                            raise ValueError(f"Address {addr_str} out of range (0x000–0xFFF)")
                        if not (0 <= value < 65536):
                            raise ValueError(f"Value {value_str} out of range (0x0000–0xFFFF)")
                        entries.append((address, value))
            except FileNotFoundError:
                # It is valid for data_file to be absent; program_file must exist
                if path != data_file:
                    raise
            return entries

        # Clear existing memory and state
        self.memory = [0] * 4096
        self.AC = self.DR = self.AR = self.IR = self.TR = 0
        self.PC = 0
        self.E = self.I = 0
        self.micro_phase = 0
        self.halted = False
        self.cycle_counter = 0
        self.instruction_counter = 0
        self.read_count = 0
        self.write_count = 0
        self.halt_pc = None
        self.last_executed_instruction = None

        # Load program and data
        program_entries = parse_file(program_file)
        data_entries: List[Tuple[int, int]] = []
        if data_file:
            data_entries = parse_file(data_file)

        # Populate memory; program entries override data entries when same address
        for address, value in data_entries:
            self.memory[address] = value & 0xFFFF
        for address, value in program_entries:
            self.memory[address] = value & 0xFFFF

        # Initialise PC to the smallest program address if program provided
        if program_entries:
            min_addr = min(addr for addr, _ in program_entries)
            self.PC = min_addr & 0xFFF
        else:
            self.PC = 0

    # ------------------------------------------------------------------
    # Execution controls
    #
    def _fetch_decode(self) -> None:
        """Internal helper: perform T0–T2 fetch and decode microsteps.

        This method is never called directly; `next_cycle` dispatches micro
        steps based on `self.micro_phase`.  The logic is split here purely
        for readability and to encapsulate the phase transitions.
        """
        pass  # not used; micro logic is within `next_cycle` directly

    def next_cycle(self) -> Tuple[str, List[str]]:
        """Execute a single clock cycle (micro operation).

        Advances the simulator by one micro phase, updates registers and
        memory as required, increments the cycle counter, and determines
        which micro operation string and list of changed components should be
        returned.  If the simulator is halted (due to a HLT instruction)
        subsequent calls to this method will have no effect.

        :returns: a tuple `(micro_operation, changed_components)` where
            *micro_operation* is a human‐readable description of the
            micro operation performed (e.g. ``"T4: DR ← M[AR]"``) and
            *changed_components* is a list of register or memory names
            modified during the cycle.
        """
        if self.halted:
            return ("Simulator halted", [])

        changed: List[str] = []
        micro_op: str = ""

        # Increment cycle count for each micro operation executed
        self.cycle_counter += 1

        # Determine which phase we're in
        phase = self.micro_phase

        # T0: AR <- PC
        if phase == 0:
            # Save old AR for changed detection
            self.AR = self.PC & 0xFFF
            changed.append('AR')
            micro_op = "T0: AR ← PC"
            self.micro_phase = 1
        elif phase == 1:
            # T1: IR <- M[AR]; PC <- PC + 1
            # Memory read
            self.IR = self.memory[self.AR] & 0xFFFF
            self.read_count += 1
            self.PC = (self.PC + 1) & 0xFFF
            changed.extend(['IR', 'PC'])
            micro_op = "T1: IR ← M[AR]; PC ← PC + 1"
            self.micro_phase = 2
        elif phase == 2:
            # T2: AR <- IR[0-11]; I <- IR[15]
            self.AR = self.IR & 0x0FFF
            self.I = 1 if (self.IR & 0x8000) else 0
            changed.extend(['AR', 'I'])
            micro_op = "T2: AR ← IR[0–11]; I ← IR[15]"
            # After decode we always proceed to phase 3
            self.micro_phase = 3
        else:
            # After T2 we are in the execution phases (T3–T6).  We
            # determine the instruction class here.
            opcode = (self.IR >> 12) & 0x7  # bits 14–12
            if opcode == 7 and self.I == 0:
                # Register reference instruction (D7 I=0): execute all
                # register operations in a single micro step.
                reg_changed: List[str] = []
                operations: List[str] = []
                # Map bits to operations
                ir = self.IR & 0x0FFF
                # Bit 11 (2048): CLA
                if ir & 0x800:  # 2048
                    self.AC = 0
                    reg_changed.append('AC')
                    operations.append('CLA')
                # Bit 10: CLE
                if ir & 0x400:  # 1024
                    self.E = 0
                    reg_changed.append('E')
                    operations.append('CLE')
                # Bit 9: CMA
                if ir & 0x200:  # 512
                    self.AC = (~self.AC) & 0xFFFF
                    reg_changed.append('AC')
                    operations.append('CMA')
                # Bit 8: CME
                if ir & 0x100:  # 256
                    self.E = self.E ^ 1
                    reg_changed.append('E')
                    operations.append('CME')
                # Bit 7: CIR – rotate right through carry
                if ir & 0x80:  # 128
                    combined = (self.E << 16) | self.AC
                    # rotate right by one: lowest bit moves to carry; carry moves to bit 15
                    new_carry = combined & 1
                    combined >>= 1
                    # Wrap around old carry bit into MSB of AC
                    if new_carry:
                        combined |= (1 << 16)
                    # Extract AC and E
                    self.E = (combined >> 16) & 1
                    self.AC = combined & 0xFFFF
                    reg_changed.extend(['AC', 'E'])
                    operations.append('CIR')
                # Bit 6: CIL – rotate left through carry
                if ir & 0x40:  # 64
                    combined = (self.AC << 1) | self.E
                    new_carry = (combined >> 16) & 1
                    combined &= 0x1FFFF  # 17 bits
                    # Wrap around old MSB of AC into LSB
                    if (combined >> 16) & 1:
                        pass  # bit already placed into carry via new_carry
                    # Extract E and AC
                    self.E = new_carry
                    self.AC = combined & 0xFFFF
                    reg_changed.extend(['AC', 'E'])
                    operations.append('CIL')
                # Bit 5: INC – increment AC
                if ir & 0x20:  # 32
                    self.AC = (self.AC + 1) & 0xFFFF
                    reg_changed.append('AC')
                    operations.append('INC')
                # Bit 4: SPA – skip next instruction if AC >= 0 (sign bit 0)
                if ir & 0x10:  # 16
                    if (self.AC & 0x8000) == 0:
                        self.PC = (self.PC + 1) & 0xFFF
                        reg_changed.append('PC')
                        operations.append('SPA (skip)')
                # Bit 3: SNA – skip next instruction if AC < 0 (sign bit 1)
                if ir & 0x8:  # 8
                    if (self.AC & 0x8000) != 0:
                        self.PC = (self.PC + 1) & 0xFFF
                        reg_changed.append('PC')
                        operations.append('SNA (skip)')
                # Bit 2: SZA – skip if AC == 0
                if ir & 0x4:  # 4
                    if self.AC == 0:
                        self.PC = (self.PC + 1) & 0xFFF
                        reg_changed.append('PC')
                        operations.append('SZA (skip)')
                # Bit 1: SZE – skip if E == 0
                if ir & 0x2:  # 2
                    if self.E == 0:
                        self.PC = (self.PC + 1) & 0xFFF
                        reg_changed.append('PC')
                        operations.append('SZE (skip)')
                # Bit 0: HLT – halt the simulator
                if ir & 0x1:  # 1
                    self.halted = True
                    self.halt_pc = self.PC
                    operations.append('HLT')
                # Compose micro operation description
                if operations:
                    micro_op = "T3: " + ", ".join(operations)
                else:
                    micro_op = "T3: NOP"
                changed.extend(reg_changed)
                # Completed instruction: record the executed instruction and reset
                self.last_executed_instruction = self.IR & 0xFFFF
                self.micro_phase = 0
                self.instruction_counter += 1
            else:
                # Memory reference instruction (D0–D6)
                inst_code = opcode
                # Execution depends on phase and indirect flag
                if phase == 3:
                    # T3: if indirect, fetch effective address; otherwise NOP
                    if self.I == 1:
                        # AR <- M[AR]
                        self.AR = self.memory[self.AR] & 0xFFFF
                        self.read_count += 1
                        self.AR &= 0x0FFF
                        changed.append('AR')
                        micro_op = "T3: AR ← M[AR] (indirect)"
                    else:
                        micro_op = "T3: (direct)"
                    self.micro_phase = 4
                elif phase == 4:
                    # T4: perform first execution step depending on instruction
                    if inst_code == 0:  # AND
                        self.DR = self.memory[self.AR] & 0xFFFF
                        self.read_count += 1
                        changed.append('DR')
                        micro_op = "T4: DR ← M[AR]"
                        self.micro_phase = 5
                    elif inst_code == 1:  # ADD
                        self.DR = self.memory[self.AR] & 0xFFFF
                        self.read_count += 1
                        changed.append('DR')
                        micro_op = "T4: DR ← M[AR]"
                        self.micro_phase = 5
                    elif inst_code == 2:  # LDA
                        self.DR = self.memory[self.AR] & 0xFFFF
                        self.read_count += 1
                        changed.append('DR')
                        micro_op = "T4: DR ← M[AR]"
                        self.micro_phase = 5
                    elif inst_code == 3:  # STA
                        # Store AC into memory
                        self.memory[self.AR] = self.AC & 0xFFFF
                        self.write_count += 1
                        changed.append(f"M[{self.AR:03X}]")
                        micro_op = "T4: M[AR] ← AC"
                        # STA ends at T4; record executed instruction and reset
                        self.last_executed_instruction = self.IR & 0xFFFF
                        self.micro_phase = 0
                        self.instruction_counter += 1
                    elif inst_code == 4:  # BUN
                        # Branch unconditionally: PC <- AR
                        self.PC = self.AR & 0xFFF
                        changed.append('PC')
                        micro_op = "T4: PC ← AR"
                        # BUN ends at T4; record executed instruction and reset
                        self.last_executed_instruction = self.IR & 0xFFFF
                        self.micro_phase = 0
                        self.instruction_counter += 1
                    elif inst_code == 5:  # BSA
                        # Store return address (PC) into memory at AR
                        self.memory[self.AR] = self.PC & 0xFFFF
                        self.write_count += 1
                        changed.append(f"M[{self.AR:03X}]")
                        micro_op = "T4: M[AR] ← PC"
                        # Next micro phase will set PC <- AR + 1
                        self.micro_phase = 5
                    elif inst_code == 6:  # ISZ
                        # Fetch memory word into DR
                        self.DR = self.memory[self.AR] & 0xFFFF
                        self.read_count += 1
                        changed.append('DR')
                        micro_op = "T4: DR ← M[AR]"
                        self.micro_phase = 5
                    else:
                        # Should never reach here; reserved
                        micro_op = "Invalid instruction"
                        self.last_executed_instruction = self.IR & 0xFFFF
                        self.micro_phase = 0
                        self.instruction_counter += 1
                elif phase == 5:
                    # T5: perform second step
                    if inst_code == 0:  # AND
                        self.AC = self.AC & self.DR
                        changed.append('AC')
                        micro_op = "T5: AC ← AC ∧ DR"
                        # End of instruction; record executed instruction
                        self.last_executed_instruction = self.IR & 0xFFFF
                        self.micro_phase = 0
                        self.instruction_counter += 1
                    elif inst_code == 1:  # ADD
                        total = self.AC + self.DR
                        self.E = 1 if total > 0xFFFF else 0
                        self.AC = total & 0xFFFF
                        changed.extend(['AC', 'E'])
                        micro_op = "T5: AC ← AC + DR; E ← carry"
                        # End of instruction; record executed instruction
                        self.last_executed_instruction = self.IR & 0xFFFF
                        self.micro_phase = 0
                        self.instruction_counter += 1
                    elif inst_code == 2:  # LDA
                        self.AC = self.DR & 0xFFFF
                        changed.append('AC')
                        micro_op = "T5: AC ← DR"
                        # End of instruction; record executed instruction
                        self.last_executed_instruction = self.IR & 0xFFFF
                        self.micro_phase = 0
                        self.instruction_counter += 1
                    elif inst_code == 5:  # BSA
                        # PC <- AR + 1
                        self.PC = (self.AR + 1) & 0xFFF
                        changed.append('PC')
                        micro_op = "T5: PC ← AR + 1"
                        # End of instruction; record executed instruction
                        self.last_executed_instruction = self.IR & 0xFFFF
                        self.micro_phase = 0
                        self.instruction_counter += 1
                    elif inst_code == 6:  # ISZ
                        # Increment DR; write back; if result zero then PC++
                        self.DR = (self.DR + 1) & 0xFFFF
                        self.memory[self.AR] = self.DR & 0xFFFF
                        self.write_count += 1
                        changed.extend(['DR', f"M[{self.AR:03X}]"])
                        micro_op = "T5: DR ← DR + 1; M[AR] ← DR"
                        self.micro_phase = 6
                    else:
                        # Should not occur
                        micro_op = "Invalid T5"
                        self.last_executed_instruction = self.IR & 0xFFFF
                        self.micro_phase = 0
                        self.instruction_counter += 1
                elif phase == 6 and inst_code == 6:
                    # T6 for ISZ: conditional skip
                    if self.DR == 0:
                        self.PC = (self.PC + 1) & 0xFFF
                        changed.append('PC')
                        micro_op = "T6: DR = 0 → PC ← PC + 1"
                    else:
                        micro_op = "T6: DR ≠ 0"
                    # End of instruction; record executed instruction
                    self.last_executed_instruction = self.IR & 0xFFFF
                    self.micro_phase = 0
                    self.instruction_counter += 1
                else:
                    # For completeness, but unreachable
                    micro_op = "Idle"
                    self.last_executed_instruction = self.IR & 0xFFFF
                    self.micro_phase = 0
                    self.instruction_counter += 1

        # Remove duplicate entries in changed list while preserving order
        seen: Dict[str, None] = {}
        unique_changed = [x for x in changed if not (x in seen or seen.setdefault(x, None))]

        return (micro_op, unique_changed)

    def fast_cycle(self, n: int) -> Tuple[str, List[str]]:
        """Execute ``n`` consecutive cycles.

        Runs ``next_cycle`` exactly ``n`` times (or until a halt occurs).
        Returns the micro operation and changed components of the last
        executed cycle.  If the simulator halts before completing ``n``
        cycles, execution stops immediately.

        :param n: number of cycles to execute
        :returns: same as :meth:`next_cycle` for the final cycle
        """
        last_op = ""
        last_changed: List[str] = []
        for _ in range(max(0, n)):
            op, ch = self.next_cycle()
            last_op, last_changed = op, ch
            if self.halted:
                break
        return (last_op, last_changed)

    def next_inst(self) -> Tuple[str, Dict[str, int]]:
        """Execute the next complete instruction.

        Advances through micro phases until a full instruction has been
        completed (i.e. the micro phase returns to T0), or until a halt
        occurs.  Returns a summary of the executed instruction and selected
        register values after execution.

        :returns: a tuple ``(inst_hex, state)`` where ``inst_hex`` is the
            hexadecimal representation of the instruction that was executed
            and ``state`` is a dictionary of key registers (currently ``AC``
            and ``PC``) after execution.  If the simulator is halted, the
            last instruction executed before halting is returned.
        """
        if self.halted:
            return ("HLT", {"AC": self.AC, "PC": self.PC})
        # Execute micro cycles until the next instruction boundary or halt
        while True:
            op, _ = self.next_cycle()
            # Once the micro phase resets to 0, an instruction has completed
            if self.micro_phase == 0:
                break
            if self.halted:
                break
        # Determine which instruction was executed.  `last_executed_instruction`
        # is set inside `next_cycle` when an instruction finishes.  If it
        # remains ``None`` (e.g. on the very first call before any fetch),
        # fall back to the current IR.
        executed = self.last_executed_instruction
        if executed is None:
            executed = self.IR
        inst_hex = f"0x{executed & 0xFFFF:04X}"
        # Capture AC and PC after execution
        state = {"AC": self.AC, "PC": self.PC}
        return (inst_hex, state)

    def fast_inst(self, n: int) -> Tuple[str, Dict[str, int]]:
        """Execute ``n`` instructions in sequence.

        Uses :meth:`next_inst` repeatedly.  Stops early if the simulator
        halts.  Returns the last executed instruction and state.

        :param n: number of instructions to execute
        :returns: same as :meth:`next_inst` for the final instruction
        """
        last_inst = ""
        last_state: Dict[str, int] = {}
        for _ in range(max(0, n)):
            inst, state = self.next_inst()
            last_inst, last_state = inst, state
            if self.halted:
                break
        return (last_inst, last_state)

    def run(self) -> Tuple[str, Dict[str, int]]:
        """Run the program until a halt occurs.

        Executes instructions continuously until a HLT instruction is
        encountered.  Returns the last executed instruction and final state.

        :returns: same as :meth:`next_inst` for the last instruction
        """
        last_inst = ""
        last_state: Dict[str, int] = {}
        while not self.halted:
            inst, state = self.next_inst()
            last_inst, last_state = inst, state
            if self.halted:
                break
        return (last_inst, last_state)

    # ------------------------------------------------------------------
    # Inspection methods
    #
    def _format_binary(self, value: int, bits: int) -> str:
        """Return a binary string grouped in fours for readability.

        :param value: integer value to format
        :param bits: number of bits to represent
        :returns: a string like ``"0000 1010 1100 0011"``
        """
        bin_str = f"{value:0{bits}b}"
        return " ".join(bin_str[i:i+4] for i in range(0, bits, 4))

    def show_register(self, name: str) -> str:
        """Return a formatted string for a register value.

        Supports ``AC``, ``DR``, ``AR``, ``PC``, ``IR``, ``TR``, ``E`` and ``I``.

        :param name: register name (case insensitive)
        :returns: formatted representation or a message if unknown
        """
        name = name.upper()
        if name == 'AC':
            return f"AC = 0x{self.AC:04X} (binary: {self._format_binary(self.AC, 16)})"
        elif name == 'DR':
            return f"DR = 0x{self.DR:04X} (binary: {self._format_binary(self.DR, 16)})"
        elif name == 'AR':
            return f"AR = 0x{self.AR:03X} (binary: {self._format_binary(self.AR, 12)})"
        elif name == 'PC':
            return f"PC = 0x{self.PC:03X} (binary: {self._format_binary(self.PC, 12)})"
        elif name == 'IR':
            return f"IR = 0x{self.IR:04X} (binary: {self._format_binary(self.IR, 16)})"
        elif name == 'TR':
            return f"TR = 0x{self.TR:04X} (binary: {self._format_binary(self.TR, 16)})"
        elif name == 'E':
            return f"E = {self.E}"
        elif name == 'I':
            return f"I = {self.I}"
        else:
            return f"Unknown register: {name}"

    def show_memory(self, addr: int, count: int = 1) -> str:
        """Return a formatted representation of memory contents.

        :param addr: starting address (integer)
        :param count: number of words to display
        :returns: string with one line per word
        """
        lines: List[str] = []
        for offset in range(max(1, count)):
            a = (addr + offset) if count > 1 else addr + offset - 0  # correct indexing for single vs multiple
            if 0 <= a < len(self.memory):
                value = self.memory[a] & 0xFFFF
                if count == 1:
                    # single address: include binary
                    bin_rep = self._format_binary(value, 16)
                    lines.append(f"M[{a}] = 0x{value:04X} (binary: {bin_rep})")
                else:
                    lines.append(f"0x{a:03X}   | 0x{value:04X}")
        return "\n".join(lines)

    def show_all(self) -> str:
        """Return a single‐line summary of all registers and the micro phase.

        Includes AC, DR, AR, PC, IR, TR, E, I and the current micro phase.
        """
        return (
            f"AC=0x{self.AC:04X}  DR=0x{self.DR:04X}  AR=0x{self.AR:03X}  "
            f"PC=0x{self.PC:03X}  IR=0x{self.IR:04X}  TR=0x{self.TR:04X}  "
            f"E={self.E}  I={self.I}  SC={self.micro_phase}"
        )

    def show_profiler(self) -> str:
        """Return profiling statistics as a formatted string."""
        inst = self.instruction_counter if self.instruction_counter > 0 else 1
        cpi = self.cycle_counter / inst
        mbw = self.read_count + self.write_count
        return (
            f"Total cycles: {self.cycle_counter}\n"
            f"Instructions executed: {self.instruction_counter}\n"
            f"Average cycles per instruction (CPI): {cpi:.2f}\n"
            f"Memory reads: {self.read_count}\n"
            f"Memory writes: {self.write_count}\n"
            f"Memory bandwidth (reads + writes): {mbw}"
        )

    # ------------------------------------------------------------------
    # Command parsing
    #
    def execute_command(self, command: str) -> str:
        """Parse and execute a CLI command.

        Accepts a string command, dispatches to the appropriate simulator
        method and returns a human‐readable response.  Unknown commands
        produce a helpful message.  Commands are case insensitive.

        Supported commands:

        * ``next_cycle`` – execute one micro cycle.
        * ``fast_cycle N`` – execute ``N`` micro cycles.
        * ``next_inst`` – execute one full instruction.
        * ``fast_inst N`` – execute ``N`` instructions.
        * ``run`` – run until halt.
        * ``show <reg>`` – display the value of a register.
        * ``show mem <addr> [count]`` – display memory contents starting at
          ``addr`` for ``count`` locations (default 1).
        * ``show all`` – display all registers and micro phase.
        * ``show profiler`` – display profiling statistics.
        * ``exit`` or ``quit`` – return an empty string to signal loop exit.

        :param command: command string entered by the user
        :returns: response string; if empty, caller should terminate the loop
        """
        cmd = command.strip()
        if not cmd:
            return ""
        tokens = cmd.split()
        base = tokens[0].lower()
        # next_cycle
        if base == 'next_cycle':
            op, changed = self.next_cycle()
            inst_in_hand = f"0x{self.IR:04X}"
            changed_str = ', '.join(changed) if changed else 'None'
            return (f"Instruction in hand: {inst_in_hand}\n"
                    f"Micro-operation: {op}\n"
                    f"Changed: {changed_str}")
        elif base == 'fast_cycle':
            if len(tokens) < 2 or not tokens[1].isdigit():
                return "Usage: fast_cycle N"
            n = int(tokens[1])
            op, changed = self.fast_cycle(n)
            inst_in_hand = f"0x{self.IR:04X}"
            changed_str = ', '.join(changed) if changed else 'None'
            return (f"Instruction in hand: {inst_in_hand}\n"
                    f"Micro-operation: {op}\n"
                    f"Changed: {changed_str}")
        elif base == 'next_inst':
            inst, state = self.next_inst()
            # Show only AC and PC after instruction
            return (f"Instruction executed: {inst}\n"
                    f"PC = 0x{state['PC']:03X}  AC = 0x{state['AC']:04X}")
        elif base == 'fast_inst':
            if len(tokens) < 2 or not tokens[1].isdigit():
                return "Usage: fast_inst N"
            n = int(tokens[1])
            inst, state = self.fast_inst(n)
            return (f"Instruction executed: {inst}\n"
                    f"PC = 0x{state['PC']:03X}  AC = 0x{state['AC']:04X}")
        elif base == 'run':
            inst, state = self.run()
            # Provide summary after run
            if self.halted:
                return (f"Program halted.\n"
                        f"Final instruction: {inst}\n"
                        f"PC = 0x{state['PC']:03X}  AC = 0x{state['AC']:04X}")
            else:
                return (f"Run completed without halt.\n"
                        f"Last instruction: {inst}\n"
                        f"PC = 0x{state['PC']:03X}  AC = 0x{state['AC']:04X}")
        elif base == 'show':
            if len(tokens) < 2:
                return "Usage: show <reg|mem|all|profiler> [args]"
            target = tokens[1].lower()
            if target == 'all':
                return self.show_all()
            elif target == 'profiler':
                return self.show_profiler()
            elif target == 'mem':
                if len(tokens) < 3:
                    return "Usage: show mem <addr> [count]"
                # Parse address (allow hex 0x.. or decimal)
                addr_str = tokens[2]
                try:
                    if addr_str.lower().startswith('0x'):
                        addr = int(addr_str, 16)
                    else:
                        addr = int(addr_str, 0)
                except ValueError:
                    return f"Invalid address: {addr_str}"
                count = 1
                if len(tokens) >= 4:
                    try:
                        count = int(tokens[3])
                    except ValueError:
                        return f"Invalid count: {tokens[3]}"
                return self.show_memory(addr, count)
            else:
                # assume register name
                return self.show_register(target)
        elif base in ('exit', 'quit'):
            # Signal caller to exit
            return ""
        else:
            return f"Unknown command: {tokens[0]}"


def main() -> None:
    """Simple command line interface for the Basic Computer Simulator.

    Upon invocation this function loads `program.txt` and `data.txt` from the
    current working directory and enters a read–eval–print loop.  The loop
    terminates when the user enters ``exit`` or ``quit``.  When executing in
    an environment without a console (e.g. imported as a module) this function
    does nothing.
    """
    import sys
    if not sys.stdin.isatty():
        # Non‑interactive environment – do not launch CLI
        return
    sim = BasicComputerSimulator()
    try:
        sim.load_program()
    except FileNotFoundError:
        print("Warning: program.txt not found. Memory initialised to zeros.\n", file=sys.stderr)
    print("Basic Computer Simulator. Type 'help' for commands.\n")
    while True:
        try:
            cmd = input('> ')
        except EOFError:
            break
        cmd = cmd.strip()
        if cmd.lower() == 'help':
            print("Available commands:\n"
                  "  next_cycle           – execute one clock cycle\n"
                  "  fast_cycle N         – execute N clock cycles\n"
                  "  next_inst            – execute one instruction\n"
                  "  fast_inst N          – execute N instructions\n"
                  "  run                  – run until halt\n"
                  "  show <reg>           – show a register (AC, DR, AR, PC, IR, TR, E, I)\n"
                  "  show mem <addr> [c]  – show memory word(s)\n"
                  "  show all             – show all registers and micro phase\n"
                  "  show profiler        – show profiling statistics\n"
                  "  exit, quit           – exit simulator")
            continue
        response = sim.execute_command(cmd)
        if response == "":
            break
        print(response)


if __name__ == '__main__':
    main()