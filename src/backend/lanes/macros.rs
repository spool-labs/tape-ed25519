//! Macros for the AVX-512 lane kernels and the eight-lane SHA-512

/// Compute schedule word `$i` in the sixteen-word ring with indices fixed at compile time
macro_rules! sched {
    ($w:expr, $i:expr) => {{
        const I: usize = $i;
        let w2 = $w[(I + 14) % 16];
        let w15 = $w[(I + 1) % 16];
        let s1 = _mm512_ternarylogic_epi64::<0x96>(
            _mm512_ror_epi64::<19>(w2),
            _mm512_ror_epi64::<61>(w2),
            _mm512_srli_epi64::<6>(w2),
        );
        let s0 = _mm512_ternarylogic_epi64::<0x96>(
            _mm512_ror_epi64::<1>(w15),
            _mm512_ror_epi64::<8>(w15),
            _mm512_srli_epi64::<7>(w15),
        );
        $w[I % 16] = _mm512_add_epi64(
            _mm512_add_epi64($w[I % 16], $w[(I + 9) % 16]),
            _mm512_add_epi64(s0, s1),
        );
    }};
}

/// Run one SHA-512 round on one group, writing only `d` and `h` so no state word moves
macro_rules! round {
    ($k:expr, $w:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident) => {{
        let big_s1 = _mm512_ternarylogic_epi64::<0x96>(
            _mm512_ror_epi64::<14>($e),
            _mm512_ror_epi64::<18>($e),
            _mm512_ror_epi64::<41>($e),
        );
        let ch = _mm512_ternarylogic_epi64::<0xCA>($e, $f, $g);
        let t1 = _mm512_add_epi64(
            _mm512_add_epi64($h, big_s1),
            _mm512_add_epi64(_mm512_add_epi64(ch, $k), $w),
        );
        let big_s0 = _mm512_ternarylogic_epi64::<0x96>(
            _mm512_ror_epi64::<28>($a),
            _mm512_ror_epi64::<34>($a),
            _mm512_ror_epi64::<39>($a),
        );
        let maj = _mm512_ternarylogic_epi64::<0xE8>($a, $b, $c);
        $d = _mm512_add_epi64($d, t1);
        $h = _mm512_add_epi64(t1, _mm512_add_epi64(big_s0, maj));
    }};
}

/// Run round `$round_base + $slot` for every listed group with one shared K load
macro_rules! round_step {
    (plain, $round_base:expr, $slot:expr
     $(, [$w:ident, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident])+) => {{
        // SAFETY: round indices stay below the eighty aligned entries of K64_X8
        let k = _mm512_load_si512(K64_X8.0.as_ptr().add($round_base + $slot) as *const _);
        $( round!(k, $w[$slot % 16], $a, $b, $c, $d, $e, $f, $g, $h); )+
    }};
    (sched, $round_base:expr, $slot:expr
     $(, [$w:ident, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident])+) => {{
        $( sched!($w, $slot + 1); )+
        // SAFETY: round indices stay below the eighty aligned entries of K64_X8
        let k = _mm512_load_si512(K64_X8.0.as_ptr().add($round_base + $slot) as *const _);
        $( round!(k, $w[$slot % 16], $a, $b, $c, $d, $e, $f, $g, $h); )+
    }};
}

/// Run eight rounds and rotate the state roles one step per round
macro_rules! rounds8 {
    ($mode:tt, $round_base:expr, $offset:expr
     $(, [$w:ident, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident])+) => {{
        round_step!($mode, $round_base, $offset + 0 $(, [$w, $a, $b, $c, $d, $e, $f, $g, $h])+);
        round_step!($mode, $round_base, $offset + 1 $(, [$w, $h, $a, $b, $c, $d, $e, $f, $g])+);
        round_step!($mode, $round_base, $offset + 2 $(, [$w, $g, $h, $a, $b, $c, $d, $e, $f])+);
        round_step!($mode, $round_base, $offset + 3 $(, [$w, $f, $g, $h, $a, $b, $c, $d, $e])+);
        round_step!($mode, $round_base, $offset + 4 $(, [$w, $e, $f, $g, $h, $a, $b, $c, $d])+);
        round_step!($mode, $round_base, $offset + 5 $(, [$w, $d, $e, $f, $g, $h, $a, $b, $c])+);
        round_step!($mode, $round_base, $offset + 6 $(, [$w, $c, $d, $e, $f, $g, $h, $a, $b])+);
        round_step!($mode, $round_base, $offset + 7 $(, [$w, $b, $c, $d, $e, $f, $g, $h, $a])+);
    }};
}

/// Run the eighty rounds of one block for every listed group
macro_rules! compress_rounds {
    ($([$w:ident, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident]),+) => {{
        rounds8!(plain, 0usize, 0 $(, [$w, $a, $b, $c, $d, $e, $f, $g, $h])+);
        rounds8!(plain, 0usize, 8 $(, [$w, $a, $b, $c, $d, $e, $f, $g, $h])+);
        // Each later round computes the schedule word for the round after it
        $( sched!($w, 16); )+
        for base in (16..80).step_by(16) {
            rounds8!(sched, base, 0 $(, [$w, $a, $b, $c, $d, $e, $f, $g, $h])+);
            rounds8!(sched, base, 8 $(, [$w, $a, $b, $c, $d, $e, $f, $g, $h])+);
        }
    }};
}

/// Load one block's schedule and unpack the state into eight role variables
macro_rules! block_in {
    ($state:expr, $block:expr, $bswap:expr, $w:ident,
     $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident) => {
        let mut $w = load_schedule(&$block, $bswap);
        let mut $a = $state[0];
        let mut $b = $state[1];
        let mut $c = $state[2];
        let mut $d = $state[3];
        let mut $e = $state[4];
        let mut $f = $state[5];
        let mut $g = $state[6];
        let mut $h = $state[7];
    };
}

/// Add the block's result into the state for every lane still inside its message
macro_rules! block_out {
    ($state:expr, $active:expr,
     $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident) => {
        $state[0] = _mm512_mask_add_epi64($state[0], $active, $state[0], $a);
        $state[1] = _mm512_mask_add_epi64($state[1], $active, $state[1], $b);
        $state[2] = _mm512_mask_add_epi64($state[2], $active, $state[2], $c);
        $state[3] = _mm512_mask_add_epi64($state[3], $active, $state[3], $d);
        $state[4] = _mm512_mask_add_epi64($state[4], $active, $state[4], $e);
        $state[5] = _mm512_mask_add_epi64($state[5], $active, $state[5], $f);
        $state[6] = _mm512_mask_add_epi64($state[6], $active, $state[6], $g);
        $state[7] = _mm512_mask_add_epi64($state[7], $active, $state[7], $h);
    };
}

macro_rules! tf {
    ($(#[$meta:meta])* fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty $body:block) => {
        $(#[$meta])*
        #[inline(always)]
        unsafe fn $name($($arg: $ty),*) -> $ret $body
    };
}

macro_rules! shl {
    ($x:expr, $n:literal) => {
        _mm512_slli_epi64($x, $n)
    };
}

macro_rules! shr {
    ($x:expr, $n:literal) => {
        _mm512_srli_epi64($x, $n)
    };
}

/// Emit the eight-lane radix-51 multiplication waves inline, limbs below 2^52 in and out
macro_rules! mul_waves_raw_emit {
    ($left:expr, $right:expr) => {{
        let x: [__m512i; 5] = $left;
        let y: [__m512i; 5] = $right;

        unsafe {
            // Accumulators for terms with weight 1
            let mut z0_1 = splat(0);
            let mut z1_1 = splat(0);
            let mut z2_1 = splat(0);
            let mut z3_1 = splat(0);
            let mut z4_1 = splat(0);
            let mut z5_1 = splat(0);
            let mut z6_1 = splat(0);
            let mut z7_1 = splat(0);
            let mut z8_1 = splat(0);

            // Accumulators for terms with weight 2
            let mut z0_2 = splat(0);
            let mut z1_2 = splat(0);
            let mut z2_2 = splat(0);
            let mut z3_2 = splat(0);
            let mut z4_2 = splat(0);
            let mut z5_2 = splat(0);
            let mut z6_2 = splat(0);
            let mut z7_2 = splat(0);
            let mut z8_2 = splat(0);
            let mut z9_2 = splat(0);

            z4_1 = madd52lo(z4_1, x[2], y[2]);
            z5_2 = madd52hi(z5_2, x[2], y[2]);
            z5_1 = madd52lo(z5_1, x[4], y[1]);
            z6_2 = madd52hi(z6_2, x[4], y[1]);
            z6_1 = madd52lo(z6_1, x[4], y[2]);
            z7_2 = madd52hi(z7_2, x[4], y[2]);
            z7_1 = madd52lo(z7_1, x[4], y[3]);
            z8_2 = madd52hi(z8_2, x[4], y[3]);

            z4_1 = madd52lo(z4_1, x[3], y[1]);
            z5_2 = madd52hi(z5_2, x[3], y[1]);
            z5_1 = madd52lo(z5_1, x[3], y[2]);
            z6_2 = madd52hi(z6_2, x[3], y[2]);
            z6_1 = madd52lo(z6_1, x[3], y[3]);
            z7_2 = madd52hi(z7_2, x[3], y[3]);
            z7_1 = madd52lo(z7_1, x[3], y[4]);
            z8_2 = madd52hi(z8_2, x[3], y[4]);

            z8_1 = madd52lo(z8_1, x[4], y[4]);
            z9_2 = madd52hi(z9_2, x[4], y[4]);
            z4_1 = madd52lo(z4_1, x[4], y[0]);
            z5_2 = madd52hi(z5_2, x[4], y[0]);
            z5_1 = madd52lo(z5_1, x[2], y[3]);
            z6_2 = madd52hi(z6_2, x[2], y[3]);
            z6_1 = madd52lo(z6_1, x[2], y[4]);
            z7_2 = madd52hi(z7_2, x[2], y[4]);

            let z8 = add(z8_1, add(z8_2, z8_2));
            let z9 = add(z9_2, z9_2);

            z3_1 = madd52lo(z3_1, x[3], y[0]);
            z4_2 = madd52hi(z4_2, x[3], y[0]);
            z4_1 = madd52lo(z4_1, x[1], y[3]);
            z5_2 = madd52hi(z5_2, x[1], y[3]);
            z5_1 = madd52lo(z5_1, x[1], y[4]);
            z6_2 = madd52hi(z6_2, x[1], y[4]);
            z2_1 = madd52lo(z2_1, x[2], y[0]);
            z3_2 = madd52hi(z3_2, x[2], y[0]);

            let z6 = add(z6_1, add(z6_2, z6_2));
            let z7 = add(z7_1, add(z7_2, z7_2));

            z3_1 = madd52lo(z3_1, x[2], y[1]);
            z4_2 = madd52hi(z4_2, x[2], y[1]);
            z4_1 = madd52lo(z4_1, x[0], y[4]);
            z5_2 = madd52hi(z5_2, x[0], y[4]);
            z1_1 = madd52lo(z1_1, x[1], y[0]);
            z2_2 = madd52hi(z2_2, x[1], y[0]);
            z2_1 = madd52lo(z2_1, x[1], y[1]);
            z3_2 = madd52hi(z3_2, x[1], y[1]);

            let z5 = add(z5_1, add(z5_2, z5_2));

            z3_1 = madd52lo(z3_1, x[1], y[2]);
            z4_2 = madd52hi(z4_2, x[1], y[2]);
            z0_1 = madd52lo(z0_1, x[0], y[0]);
            z1_2 = madd52hi(z1_2, x[0], y[0]);
            z1_1 = madd52lo(z1_1, x[0], y[1]);
            z2_1 = madd52lo(z2_1, x[0], y[2]);
            z2_2 = madd52hi(z2_2, x[0], y[1]);
            z3_2 = madd52hi(z3_2, x[0], y[2]);

            let mut t0 = splat(0);
            let mut t1 = splat(0);
            let r19 = splat(19);

            t0 = madd52hi(t0, r19, z9);
            t1 = madd52lo(t1, r19, shr!(z9, 52));
            z3_1 = madd52lo(z3_1, x[0], y[3]);
            z4_2 = madd52hi(z4_2, x[0], y[3]);
            z1_2 = madd52lo(z1_2, r19, shr!(z5, 52));
            z2_2 = madd52lo(z2_2, r19, shr!(z6, 52));
            z3_2 = madd52lo(z3_2, r19, shr!(z7, 52));
            z0_1 = madd52lo(z0_1, r19, z5);

            z4_1 = madd52lo(z4_1, r19, z9);
            z1_1 = madd52lo(z1_1, r19, z6);
            z0_2 = madd52lo(z0_2, r19, add(t0, t1));
            z4_2 = madd52hi(z4_2, r19, z8);
            z2_1 = madd52lo(z2_1, r19, z7);
            z1_2 = madd52hi(z1_2, r19, z5);
            z2_2 = madd52hi(z2_2, r19, z6);
            z3_2 = madd52hi(z3_2, r19, z7);

            z3_1 = madd52lo(z3_1, r19, z8);
            z4_2 = madd52lo(z4_2, r19, shr!(z8, 52));

            [
                add(z0_1, add(z0_2, z0_2)),
                add(z1_1, add(z1_2, z1_2)),
                add(z2_1, add(z2_2, z2_2)),
                add(z3_1, add(z3_2, z3_2)),
                add(z4_1, add(z4_2, z4_2)),
            ]
        }
    }};
}

/// Emit the eight-lane radix-51 squaring waves inline, limbs below 2^52 in and out
macro_rules! square_waves_raw_emit {
    ($input:expr) => {{
        let x: [__m512i; 5] = $input;

        unsafe {
            // Accumulators for terms with weight 2
            let mut z1_2 = splat(0);
            let mut z2_2 = splat(0);
            let mut z3_2 = splat(0);
            let mut z4_2 = splat(0);
            let mut z5_2 = splat(0);
            let mut z6_2 = splat(0);
            let mut z7_2 = splat(0);
            let mut z9_2 = splat(0);

            // Accumulators for terms with weight 4
            let mut z2_4 = splat(0);
            let mut z3_4 = splat(0);
            let mut z4_4 = splat(0);
            let mut z5_4 = splat(0);
            let mut z6_4 = splat(0);
            let mut z7_4 = splat(0);
            let mut z8_4 = splat(0);

            let mut z0_1 = splat(0);
            z0_1 = madd52lo(z0_1, x[0], x[0]);

            let mut z1_1 = splat(0);
            z1_2 = madd52lo(z1_2, x[0], x[1]);
            z1_2 = madd52hi(z1_2, x[0], x[0]);

            z2_4 = madd52hi(z2_4, x[0], x[1]);
            let mut z2_1 = shl!(z2_4, 2);
            z2_2 = madd52lo(z2_2, x[0], x[2]);
            z2_1 = madd52lo(z2_1, x[1], x[1]);

            z3_4 = madd52hi(z3_4, x[0], x[2]);
            let mut z3_1 = shl!(z3_4, 2);
            z3_2 = madd52lo(z3_2, x[1], x[2]);
            z3_2 = madd52lo(z3_2, x[0], x[3]);
            z3_2 = madd52hi(z3_2, x[1], x[1]);

            z4_4 = madd52hi(z4_4, x[1], x[2]);
            z4_4 = madd52hi(z4_4, x[0], x[3]);
            let mut z4_1 = shl!(z4_4, 2);
            z4_2 = madd52lo(z4_2, x[1], x[3]);
            z4_2 = madd52lo(z4_2, x[0], x[4]);
            z4_1 = madd52lo(z4_1, x[2], x[2]);

            z5_4 = madd52hi(z5_4, x[1], x[3]);
            z5_4 = madd52hi(z5_4, x[0], x[4]);
            let mut z5_1 = shl!(z5_4, 2);
            z5_2 = madd52lo(z5_2, x[2], x[3]);
            z5_2 = madd52lo(z5_2, x[1], x[4]);
            z5_2 = madd52hi(z5_2, x[2], x[2]);

            z6_4 = madd52hi(z6_4, x[2], x[3]);
            z6_4 = madd52hi(z6_4, x[1], x[4]);
            let mut z6_1 = shl!(z6_4, 2);
            z6_2 = madd52lo(z6_2, x[2], x[4]);
            z6_1 = madd52lo(z6_1, x[3], x[3]);

            z7_4 = madd52hi(z7_4, x[2], x[4]);
            let mut z7_1 = shl!(z7_4, 2);
            z7_2 = madd52lo(z7_2, x[3], x[4]);
            z7_2 = madd52hi(z7_2, x[3], x[3]);

            z8_4 = madd52hi(z8_4, x[3], x[4]);
            let mut z8_1 = shl!(z8_4, 2);
            z8_1 = madd52lo(z8_1, x[4], x[4]);

            let mut z9_1 = splat(0);
            z9_2 = madd52hi(z9_2, x[4], x[4]);

            z5_1 = add(z5_1, shl!(z5_2, 1));
            z6_1 = add(z6_1, shl!(z6_2, 1));
            z7_1 = add(z7_1, shl!(z7_2, 1));
            z9_1 = add(z9_1, shl!(z9_2, 1));

            let mut t0 = splat(0);
            let mut t1 = splat(0);
            let r19 = splat(19);

            t0 = madd52hi(t0, r19, z9_1);
            t1 = madd52lo(t1, r19, shr!(z9_1, 52));

            z4_2 = madd52lo(z4_2, r19, shr!(z8_1, 52));
            z3_2 = madd52lo(z3_2, r19, shr!(z7_1, 52));
            z2_2 = madd52lo(z2_2, r19, shr!(z6_1, 52));
            z1_2 = madd52lo(z1_2, r19, shr!(z5_1, 52));

            let mut z0_2 = splat(0);
            z0_2 = madd52lo(z0_2, r19, add(t0, t1));
            z1_2 = madd52hi(z1_2, r19, z5_1);
            z2_2 = madd52hi(z2_2, r19, z6_1);
            z3_2 = madd52hi(z3_2, r19, z7_1);
            z4_2 = madd52hi(z4_2, r19, z8_1);

            z0_1 = madd52lo(z0_1, r19, z5_1);
            z1_1 = madd52lo(z1_1, r19, z6_1);
            z2_1 = madd52lo(z2_1, r19, z7_1);
            z3_1 = madd52lo(z3_1, r19, z8_1);
            z4_1 = madd52lo(z4_1, r19, z9_1);

            [
                add(z0_1, add(z0_2, z0_2)),
                add(z1_1, add(z1_2, z1_2)),
                add(z2_1, add(z2_2, z2_2)),
                add(z3_1, add(z3_2, z3_2)),
                add(z4_1, add(z4_2, z4_2)),
            ]
        }
    }};
}
