from math import inf

import torch
import torch.fx as fx
from torch import device
from torch import tensor
from torch.fx.experimental.proxy_tensor import make_fx

from torchdynamo.testing import rand_strided
from torchinductor.compile_fx import compile_fx_inner


class Repro(torch.nn.Module):
    def forward(
        self,
        primals_1,
        primals_2,
        primals_3,
        primals_4,
        primals_5,
        primals_6,
        primals_7,
        primals_8,
        primals_9,
        primals_10,
        primals_11,
        primals_12,
        primals_13,
        primals_14,
        primals_15,
        primals_16,
        primals_17,
        primals_18,
        primals_19,
        primals_20,
        primals_21,
        primals_22,
        primals_23,
        primals_24,
        primals_25,
        primals_26,
        primals_27,
        primals_28,
        primals_29,
        primals_30,
        primals_31,
        primals_32,
        primals_33,
        primals_34,
        primals_35,
        primals_36,
        primals_37,
        primals_38,
        primals_39,
        primals_40,
        primals_41,
        primals_42,
        primals_43,
        primals_44,
        primals_45,
        primals_46,
        primals_47,
        primals_48,
        primals_49,
        primals_50,
        primals_51,
        primals_52,
        primals_53,
        primals_54,
        primals_55,
        primals_56,
        primals_57,
        primals_58,
        primals_59,
        primals_60,
        primals_61,
        primals_62,
        primals_63,
        primals_64,
        primals_65,
        primals_66,
        primals_67,
        primals_68,
        primals_69,
        primals_70,
        primals_71,
        primals_72,
        primals_73,
        primals_74,
        primals_75,
        primals_76,
        primals_77,
        primals_78,
        primals_79,
        primals_80,
        primals_81,
        primals_82,
        primals_83,
        primals_84,
        primals_85,
        primals_86,
        primals_87,
        primals_88,
        primals_89,
        primals_90,
        primals_91,
        primals_92,
        primals_93,
        primals_94,
        primals_95,
        primals_96,
        primals_97,
        primals_98,
        primals_99,
        primals_100,
        primals_101,
        primals_102,
        primals_103,
        primals_104,
        primals_105,
        primals_106,
        primals_107,
        primals_108,
        primals_109,
        primals_110,
        primals_111,
        primals_112,
        primals_113,
        primals_114,
        primals_115,
        primals_116,
        primals_117,
        primals_118,
        primals_119,
        primals_120,
        primals_121,
        primals_122,
        primals_123,
        primals_124,
        primals_125,
        primals_126,
        primals_127,
        primals_128,
        primals_129,
        primals_130,
        primals_131,
        primals_132,
        primals_133,
        primals_134,
        primals_135,
        primals_136,
        primals_137,
        primals_138,
        primals_139,
        primals_140,
        primals_141,
        primals_142,
        primals_143,
        primals_144,
        primals_145,
        primals_146,
        primals_147,
        primals_148,
        primals_149,
        primals_150,
        primals_151,
        primals_152,
        primals_153,
        primals_154,
        primals_155,
        primals_156,
        primals_157,
        primals_158,
        primals_159,
        primals_160,
        primals_161,
        primals_162,
        primals_163,
        primals_164,
        primals_165,
        primals_166,
        primals_167,
        primals_168,
        primals_169,
        primals_170,
        primals_171,
        primals_172,
        primals_173,
        primals_174,
        primals_175,
        primals_176,
        primals_177,
        primals_178,
        primals_179,
        primals_180,
        primals_181,
        primals_182,
        primals_183,
        primals_184,
        primals_185,
        primals_186,
        primals_187,
        primals_188,
        primals_189,
        primals_190,
        primals_191,
        primals_192,
        primals_193,
        primals_194,
        primals_195,
        primals_196,
        primals_197,
        primals_198,
        primals_199,
        primals_200,
        primals_201,
        primals_202,
        primals_203,
        primals_204,
        primals_205,
        primals_206,
        primals_207,
        primals_208,
        primals_209,
        primals_210,
        primals_211,
        primals_212,
        primals_213,
        primals_214,
        primals_215,
        primals_216,
        primals_217,
        primals_218,
        primals_219,
        primals_220,
        primals_221,
        primals_222,
        primals_223,
        primals_224,
        primals_225,
        primals_226,
        primals_227,
        primals_228,
        primals_229,
        primals_230,
        primals_231,
        primals_232,
        primals_233,
        primals_234,
        primals_235,
        primals_236,
        primals_237,
        primals_238,
        primals_239,
        primals_240,
        primals_241,
        primals_242,
        primals_243,
        primals_244,
        primals_245,
        primals_246,
        primals_247,
        primals_248,
        primals_249,
        primals_250,
        primals_251,
        primals_252,
        primals_253,
        primals_254,
        primals_255,
        primals_256,
        primals_257,
        primals_258,
        primals_259,
        primals_260,
        primals_261,
        primals_262,
        primals_263,
        primals_264,
        primals_265,
        primals_266,
        primals_267,
        primals_268,
        primals_269,
        primals_270,
        primals_271,
        primals_272,
        primals_273,
        primals_274,
        primals_275,
        primals_276,
        primals_277,
        primals_278,
        primals_279,
        primals_280,
        primals_281,
        primals_282,
        primals_283,
        primals_284,
        primals_285,
        primals_286,
        primals_287,
        primals_288,
        primals_289,
        primals_290,
        primals_291,
        primals_292,
        primals_293,
        primals_294,
        primals_295,
        primals_296,
        primals_297,
        primals_298,
        primals_299,
        primals_300,
        primals_301,
        primals_302,
        primals_303,
        primals_304,
        primals_305,
        primals_306,
        primals_307,
        primals_308,
        primals_309,
        primals_310,
        primals_311,
        primals_312,
        primals_313,
        primals_314,
        primals_315,
        primals_316,
        primals_317,
        primals_318,
        primals_319,
        primals_320,
        primals_321,
        primals_322,
        primals_323,
        primals_324,
        primals_325,
        primals_326,
        primals_327,
        primals_328,
        primals_329,
        primals_330,
        primals_331,
        primals_332,
        primals_333,
        primals_334,
        primals_335,
        primals_336,
        primals_337,
        primals_338,
        primals_339,
        primals_340,
        primals_341,
        primals_342,
        primals_343,
        primals_344,
        primals_345,
        primals_346,
        primals_347,
        primals_348,
        primals_349,
        primals_350,
        primals_351,
        primals_352,
        primals_353,
        primals_354,
        primals_355,
        primals_356,
        primals_357,
        primals_358,
        primals_359,
        primals_360,
        primals_361,
        primals_362,
        primals_363,
        primals_364,
        primals_365,
        primals_366,
        primals_367,
        primals_368,
        primals_369,
        primals_370,
        primals_371,
        primals_372,
        primals_373,
        primals_374,
        primals_375,
        primals_376,
        primals_377,
        primals_378,
        primals_379,
        primals_380,
        primals_381,
        primals_382,
        primals_383,
        primals_384,
        primals_385,
        primals_386,
        primals_387,
        primals_388,
        primals_389,
        primals_390,
        primals_391,
        primals_392,
        primals_393,
        primals_394,
        primals_395,
        primals_396,
        primals_397,
        primals_398,
        primals_399,
        primals_400,
        primals_401,
        primals_402,
        primals_403,
        primals_404,
        primals_405,
        primals_406,
        primals_407,
        primals_408,
        primals_409,
        primals_410,
        primals_411,
        primals_412,
        primals_413,
        primals_414,
        primals_415,
        primals_416,
        primals_417,
        primals_418,
        primals_419,
        primals_420,
        primals_421,
        primals_422,
        primals_423,
        primals_424,
        primals_425,
        primals_426,
        primals_427,
        primals_428,
        primals_429,
        primals_430,
        primals_431,
        primals_432,
        primals_433,
        primals_434,
        primals_435,
        primals_436,
        primals_437,
        primals_438,
        primals_439,
        primals_440,
        primals_441,
        primals_442,
        primals_443,
        primals_444,
        primals_445,
        primals_446,
        primals_447,
        primals_448,
        primals_449,
        primals_450,
        primals_451,
        primals_452,
        primals_453,
        primals_454,
        primals_455,
        primals_456,
        primals_457,
        primals_458,
        primals_459,
        primals_460,
        primals_461,
        primals_462,
        primals_463,
        primals_464,
        primals_465,
        primals_466,
        primals_467,
        primals_468,
        primals_469,
        primals_470,
        primals_471,
        primals_472,
        primals_473,
        primals_474,
        primals_475,
        primals_476,
        primals_477,
        primals_478,
        primals_479,
        primals_480,
        primals_481,
        primals_482,
        primals_483,
        primals_484,
        primals_485,
        primals_486,
        primals_487,
        primals_488,
        primals_489,
        primals_490,
        primals_491,
        primals_492,
        primals_493,
        primals_494,
        primals_495,
        primals_496,
        primals_497,
        primals_498,
        primals_499,
        primals_500,
        primals_501,
        primals_502,
        primals_503,
        primals_504,
        primals_505,
        primals_506,
        primals_507,
        primals_508,
        primals_509,
        primals_510,
        primals_511,
        primals_512,
        primals_513,
        primals_514,
        primals_515,
        primals_516,
        primals_517,
        primals_518,
        primals_519,
        primals_520,
        primals_521,
        primals_522,
        primals_523,
        primals_524,
        primals_525,
        primals_526,
        primals_527,
        primals_528,
        primals_529,
        primals_530,
        primals_531,
        primals_532,
        primals_533,
        primals_534,
        primals_535,
        primals_536,
        primals_537,
        primals_538,
        primals_539,
        primals_540,
        primals_541,
        primals_542,
        primals_543,
        primals_544,
        primals_545,
        primals_546,
        primals_547,
        primals_548,
        primals_549,
        primals_550,
        primals_551,
        primals_552,
        primals_553,
        primals_554,
        primals_555,
        primals_556,
        primals_557,
        primals_558,
        primals_559,
        primals_560,
        primals_561,
        primals_562,
        primals_563,
        primals_564,
        primals_565,
        primals_566,
        primals_567,
        primals_568,
        primals_569,
        primals_570,
        primals_571,
        primals_572,
        primals_573,
        primals_574,
        primals_575,
        primals_576,
        primals_577,
        primals_578,
        primals_579,
        primals_580,
        primals_581,
        primals_582,
        primals_583,
        primals_584,
        primals_585,
        primals_586,
        primals_587,
        primals_588,
        primals_589,
        primals_590,
        primals_591,
        primals_592,
        primals_593,
        primals_594,
        primals_595,
        primals_596,
        primals_597,
        primals_598,
        primals_599,
        primals_600,
        primals_601,
        primals_602,
        primals_603,
        primals_604,
        primals_605,
        primals_606,
        primals_607,
        primals_608,
        primals_609,
        primals_610,
        primals_611,
        primals_612,
        primals_613,
        primals_614,
        primals_615,
        primals_616,
        primals_617,
        primals_618,
        primals_619,
        primals_620,
        primals_621,
        primals_622,
        primals_623,
        primals_624,
        primals_625,
        primals_626,
        primals_627,
        primals_628,
        primals_629,
        primals_630,
        primals_631,
        primals_632,
        primals_633,
        primals_634,
        primals_635,
        primals_636,
        primals_637,
        primals_638,
        primals_639,
        primals_640,
        primals_641,
        primals_642,
        primals_643,
        primals_644,
        primals_645,
        primals_646,
        primals_647,
        primals_648,
        primals_649,
        primals_650,
        primals_651,
        primals_652,
        primals_653,
        primals_654,
        primals_655,
        primals_656,
        primals_657,
        primals_658,
        primals_659,
        primals_660,
        primals_661,
        primals_662,
        primals_663,
        primals_664,
        primals_665,
        primals_666,
        primals_667,
        primals_668,
        primals_669,
        primals_670,
        primals_671,
        primals_672,
        primals_673,
        primals_674,
        primals_675,
        primals_676,
        primals_677,
        primals_678,
        primals_679,
        primals_680,
        primals_681,
        primals_682,
        primals_683,
        primals_684,
        primals_685,
        primals_686,
        primals_687,
        primals_688,
        primals_689,
        primals_690,
        primals_691,
        primals_692,
        primals_693,
        primals_694,
        primals_695,
        primals_696,
        primals_697,
        primals_698,
        primals_699,
        primals_700,
        primals_701,
        primals_702,
        primals_703,
        primals_704,
        primals_705,
        primals_706,
        primals_707,
        primals_708,
        primals_709,
        primals_710,
        primals_711,
        primals_712,
        primals_713,
        primals_714,
        primals_715,
        primals_716,
        primals_717,
        primals_718,
        primals_719,
        primals_720,
        primals_721,
        primals_722,
        primals_723,
        primals_724,
        primals_725,
        primals_726,
        primals_727,
        primals_728,
        primals_729,
        primals_730,
        primals_731,
        primals_732,
        primals_733,
        primals_734,
        primals_735,
        primals_736,
        primals_737,
        primals_738,
        primals_739,
        primals_740,
        primals_741,
        primals_742,
        primals_743,
        primals_744,
        primals_745,
        primals_746,
        primals_747,
        primals_748,
        primals_749,
        primals_750,
        primals_751,
        primals_752,
        primals_753,
        primals_754,
        primals_755,
        primals_756,
        primals_757,
        primals_758,
        primals_759,
        primals_760,
        primals_761,
        primals_762,
        primals_763,
        primals_764,
        primals_765,
        primals_766,
        primals_767,
        primals_768,
        primals_769,
        primals_770,
        primals_771,
        primals_772,
        primals_773,
        primals_774,
        primals_775,
        primals_776,
        primals_777,
        primals_778,
        primals_779,
        primals_780,
        primals_781,
        primals_782,
        primals_783,
        primals_784,
        primals_785,
        primals_786,
        primals_787,
        primals_788,
        primals_789,
        primals_790,
        primals_791,
        primals_792,
        primals_793,
        primals_794,
        primals_795,
        primals_796,
        primals_797,
        primals_798,
        primals_799,
        primals_800,
        primals_801,
        primals_802,
        primals_803,
        primals_804,
        primals_805,
        primals_806,
        primals_807,
        primals_808,
        primals_809,
        primals_810,
        primals_811,
        primals_812,
        primals_813,
        primals_814,
        primals_815,
        primals_816,
        primals_817,
        primals_818,
        primals_819,
        primals_820,
        primals_821,
        primals_822,
        primals_823,
        primals_824,
        primals_825,
        primals_826,
        primals_827,
        primals_828,
        primals_829,
        primals_830,
        primals_831,
        primals_832,
        primals_833,
        primals_834,
        primals_835,
        primals_836,
        primals_837,
        primals_838,
        primals_839,
        primals_840,
        primals_841,
        primals_842,
        primals_843,
        primals_844,
        primals_845,
        primals_846,
        primals_847,
        primals_848,
        primals_849,
        primals_850,
        primals_851,
        primals_852,
        primals_853,
        primals_854,
        primals_855,
        primals_856,
        primals_857,
        primals_858,
        primals_859,
        primals_860,
        primals_861,
        primals_862,
        primals_863,
        primals_864,
        primals_865,
        primals_866,
        primals_867,
        primals_868,
        primals_869,
        primals_870,
        primals_871,
        primals_872,
        primals_873,
        primals_874,
        primals_875,
        primals_876,
        primals_877,
        primals_878,
        primals_879,
        primals_880,
        primals_881,
        primals_882,
        primals_883,
        primals_884,
        primals_885,
        primals_886,
        primals_887,
        primals_888,
        primals_889,
        primals_890,
        primals_891,
        primals_892,
        primals_893,
        primals_894,
        primals_895,
        primals_896,
        primals_897,
        primals_898,
        primals_899,
        primals_900,
        primals_901,
        primals_902,
        primals_903,
        primals_904,
        primals_905,
        primals_906,
        primals_907,
        primals_908,
        primals_909,
        primals_910,
        primals_911,
        primals_912,
        primals_913,
        primals_914,
        primals_915,
        primals_916,
        primals_917,
        primals_918,
        primals_919,
        primals_920,
        primals_921,
        primals_922,
        primals_923,
        primals_924,
        primals_925,
        primals_926,
        primals_927,
        primals_928,
        primals_929,
        primals_930,
        primals_931,
        primals_932,
        primals_933,
        primals_934,
        primals_935,
        primals_936,
        primals_937,
        primals_938,
        primals_939,
        primals_940,
        primals_941,
        primals_942,
        primals_943,
        primals_944,
        primals_945,
        primals_946,
        primals_947,
        primals_948,
        primals_949,
        primals_950,
        primals_951,
        primals_952,
        primals_953,
        primals_954,
        primals_955,
        primals_956,
        primals_957,
        primals_958,
        primals_959,
        primals_960,
        primals_961,
        primals_962,
        primals_963,
        primals_964,
        primals_965,
        primals_966,
        primals_967,
        primals_968,
        primals_969,
        primals_970,
        primals_971,
        primals_972,
        primals_973,
        primals_974,
        primals_975,
        primals_976,
        primals_977,
        primals_978,
        primals_979,
        primals_980,
        primals_981,
        primals_982,
        primals_983,
        primals_984,
        primals_985,
        primals_986,
        primals_987,
        primals_988,
        primals_989,
        primals_990,
        primals_991,
        primals_992,
        primals_993,
        primals_994,
        primals_995,
        primals_996,
        primals_997,
        primals_998,
        primals_999,
        primals_1000,
        primals_1001,
        primals_1002,
        primals_1003,
        primals_1004,
        primals_1005,
        primals_1006,
        primals_1007,
        primals_1008,
        primals_1009,
        primals_1010,
        primals_1011,
        primals_1012,
        primals_1013,
        primals_1014,
        primals_1015,
        primals_1016,
        primals_1017,
        primals_1018,
        primals_1019,
        primals_1020,
        primals_1021,
        primals_1022,
        primals_1023,
        primals_1024,
        primals_1025,
        primals_1026,
        primals_1027,
        primals_1028,
        primals_1029,
        primals_1030,
        primals_1031,
        primals_1032,
        primals_1033,
        primals_1034,
        primals_1035,
        primals_1036,
        primals_1037,
        primals_1038,
        primals_1039,
        primals_1040,
        primals_1041,
        primals_1042,
        primals_1043,
        primals_1044,
        primals_1045,
        primals_1046,
        primals_1047,
        primals_1048,
        primals_1049,
        primals_1050,
        primals_1051,
        primals_1052,
        primals_1053,
        primals_1054,
        primals_1055,
        primals_1056,
        primals_1057,
        primals_1058,
        primals_1059,
        primals_1060,
        primals_1061,
        primals_1062,
        primals_1063,
        primals_1064,
        primals_1065,
        primals_1066,
        primals_1067,
        primals_1068,
        primals_1069,
        primals_1070,
        primals_1071,
        primals_1072,
        primals_1073,
        primals_1074,
        primals_1075,
        primals_1076,
        primals_1077,
        primals_1078,
        primals_1079,
        primals_1080,
        primals_1081,
        primals_1082,
        primals_1083,
        primals_1084,
        primals_1085,
        primals_1086,
        primals_1087,
        primals_1088,
        primals_1089,
        primals_1090,
        primals_1091,
        primals_1092,
        primals_1093,
        primals_1094,
        primals_1095,
        primals_1096,
        primals_1097,
        primals_1098,
        primals_1099,
        primals_1100,
        primals_1101,
        primals_1102,
        primals_1103,
        primals_1104,
        primals_1105,
        primals_1106,
        primals_1107,
        primals_1108,
        primals_1109,
        primals_1110,
        primals_1111,
        primals_1112,
        primals_1113,
        primals_1114,
        primals_1115,
        primals_1116,
        primals_1117,
        primals_1118,
        primals_1119,
        primals_1120,
        primals_1121,
        primals_1122,
        primals_1123,
        primals_1124,
        primals_1125,
        primals_1126,
        primals_1127,
        primals_1128,
        primals_1129,
        primals_1130,
        primals_1131,
        primals_1132,
        primals_1133,
        primals_1134,
        primals_1135,
        primals_1136,
        primals_1137,
        primals_1138,
        primals_1139,
        primals_1140,
        primals_1141,
        primals_1142,
        primals_1143,
        primals_1144,
        primals_1145,
        primals_1146,
        primals_1147,
        primals_1148,
        primals_1149,
        primals_1150,
        primals_1151,
        primals_1152,
        primals_1153,
        primals_1154,
        primals_1155,
        primals_1156,
        primals_1157,
        primals_1158,
        primals_1159,
        primals_1160,
        primals_1161,
        primals_1162,
        primals_1163,
        primals_1164,
        primals_1165,
        primals_1166,
        primals_1167,
        primals_1168,
        primals_1169,
        primals_1170,
        primals_1171,
        primals_1172,
        primals_1173,
        primals_1174,
        primals_1175,
        primals_1176,
        primals_1177,
        primals_1178,
        primals_1179,
        primals_1180,
        primals_1181,
        primals_1182,
        primals_1183,
        primals_1184,
        primals_1185,
        primals_1186,
        primals_1187,
        primals_1188,
        primals_1189,
        primals_1190,
        primals_1191,
        primals_1192,
        primals_1193,
        primals_1194,
        primals_1195,
        primals_1196,
        primals_1197,
        primals_1198,
        primals_1199,
        primals_1200,
        primals_1201,
        primals_1202,
        primals_1203,
        primals_1204,
        primals_1205,
        primals_1206,
        primals_1207,
        primals_1208,
        primals_1209,
        primals_1210,
        primals_1211,
        primals_1212,
        primals_1213,
        primals_1214,
        primals_1215,
        primals_1216,
        primals_1217,
        primals_1218,
        primals_1219,
        primals_1220,
        primals_1221,
        primals_1222,
        primals_1223,
        primals_1224,
        primals_1225,
        primals_1226,
        primals_1227,
        primals_1228,
        primals_1229,
        primals_1230,
        primals_1231,
        primals_1232,
        primals_1233,
        primals_1234,
        primals_1235,
        primals_1236,
        primals_1237,
        primals_1238,
        primals_1239,
        primals_1240,
        primals_1241,
        primals_1242,
        primals_1243,
        primals_1244,
        primals_1245,
        primals_1246,
        primals_1247,
        primals_1248,
        primals_1249,
        primals_1250,
        primals_1251,
        primals_1252,
        primals_1253,
        primals_1254,
        primals_1255,
        primals_1256,
        primals_1257,
        primals_1258,
        primals_1259,
        primals_1260,
        primals_1261,
        primals_1262,
        primals_1263,
        primals_1264,
        primals_1265,
        primals_1266,
        primals_1267,
        primals_1268,
        primals_1269,
        primals_1270,
        primals_1271,
        primals_1272,
        primals_1273,
        primals_1274,
        primals_1275,
        primals_1276,
        primals_1277,
        primals_1278,
        primals_1279,
        primals_1280,
        primals_1281,
        primals_1282,
        primals_1283,
        primals_1284,
        primals_1285,
        primals_1286,
        primals_1287,
        primals_1288,
        primals_1289,
        primals_1290,
        primals_1291,
        primals_1292,
        primals_1293,
        primals_1294,
        primals_1295,
        primals_1296,
        primals_1297,
        primals_1298,
        primals_1299,
        primals_1300,
        primals_1301,
        primals_1302,
        primals_1303,
        primals_1304,
        primals_1305,
        primals_1306,
        primals_1307,
        primals_1308,
        primals_1309,
        primals_1310,
        primals_1311,
        primals_1312,
        primals_1313,
        primals_1314,
        primals_1315,
        primals_1316,
        primals_1317,
        primals_1318,
        primals_1319,
        primals_1320,
        primals_1321,
        primals_1322,
        primals_1323,
        primals_1324,
        primals_1325,
        primals_1326,
        primals_1327,
        primals_1328,
        primals_1329,
        primals_1330,
        primals_1331,
        primals_1332,
        primals_1333,
        primals_1334,
        primals_1335,
        primals_1336,
        primals_1337,
        primals_1338,
        primals_1339,
        primals_1340,
        primals_1341,
        primals_1342,
        primals_1343,
        primals_1344,
        primals_1345,
        primals_1346,
        primals_1347,
        primals_1348,
        primals_1349,
        primals_1350,
        primals_1351,
        primals_1352,
        primals_1353,
        primals_1354,
        primals_1355,
        primals_1356,
        primals_1357,
        primals_1358,
        primals_1359,
        primals_1360,
        primals_1361,
        primals_1362,
        primals_1363,
        primals_1364,
        primals_1365,
        primals_1366,
        primals_1367,
        primals_1368,
        primals_1369,
        primals_1370,
        primals_1371,
        primals_1372,
        primals_1373,
        primals_1374,
        primals_1375,
        primals_1376,
        primals_1377,
        primals_1378,
        primals_1379,
        primals_1380,
        primals_1381,
        primals_1382,
        primals_1383,
        primals_1384,
        primals_1385,
        primals_1386,
        primals_1387,
        primals_1388,
        primals_1389,
        primals_1390,
        primals_1391,
        primals_1392,
        primals_1393,
        primals_1394,
        primals_1395,
        primals_1396,
        primals_1397,
        primals_1398,
        primals_1399,
        primals_1400,
        primals_1401,
        primals_1402,
        primals_1403,
        primals_1404,
        primals_1405,
        primals_1406,
        primals_1407,
        primals_1408,
        primals_1409,
        primals_1410,
        primals_1411,
        primals_1412,
        primals_1413,
        primals_1414,
        primals_1415,
        primals_1416,
        primals_1417,
        primals_1418,
        primals_1419,
        primals_1420,
        primals_1421,
        primals_1422,
        primals_1423,
        primals_1424,
        primals_1425,
        primals_1426,
        primals_1427,
        primals_1428,
        primals_1429,
        primals_1430,
        primals_1431,
        primals_1432,
        primals_1433,
        primals_1434,
        primals_1435,
        primals_1436,
        primals_1437,
        primals_1438,
        primals_1439,
        primals_1440,
        primals_1441,
        primals_1442,
        primals_1443,
        primals_1444,
        primals_1445,
        primals_1446,
        primals_1447,
        primals_1448,
        primals_1449,
        primals_1450,
        primals_1451,
        primals_1452,
        primals_1453,
        primals_1454,
        primals_1455,
        primals_1456,
        primals_1457,
        primals_1458,
        primals_1459,
        primals_1460,
        primals_1461,
        primals_1462,
        primals_1463,
        primals_1464,
        primals_1465,
        primals_1466,
        primals_1467,
        primals_1468,
        primals_1469,
        primals_1470,
        primals_1471,
        primals_1472,
        primals_1473,
        primals_1474,
        primals_1475,
        primals_1476,
        primals_1477,
        primals_1478,
        primals_1479,
        primals_1480,
        primals_1481,
        primals_1482,
        primals_1483,
        primals_1484,
        primals_1485,
        primals_1486,
        primals_1487,
        primals_1488,
        primals_1489,
        primals_1490,
        primals_1491,
        primals_1492,
        primals_1493,
        primals_1494,
        primals_1495,
        primals_1496,
        primals_1497,
        primals_1498,
        primals_1499,
        primals_1500,
        primals_1501,
        primals_1502,
        primals_1503,
        primals_1504,
        primals_1505,
        primals_1506,
        primals_1507,
        primals_1508,
        primals_1509,
        primals_1510,
        primals_1511,
        primals_1512,
        primals_1513,
        primals_1514,
        primals_1515,
        primals_1516,
        primals_1517,
        primals_1518,
        primals_1519,
        primals_1520,
        primals_1521,
        primals_1522,
        primals_1523,
        primals_1524,
        primals_1525,
        primals_1526,
        primals_1527,
        primals_1528,
        primals_1529,
        primals_1530,
        primals_1531,
        primals_1532,
        primals_1533,
        primals_1534,
        primals_1535,
        primals_1536,
        primals_1537,
        primals_1538,
        primals_1539,
        primals_1540,
        primals_1541,
        primals_1542,
        primals_1543,
        primals_1544,
        primals_1545,
        primals_1546,
        primals_1547,
        primals_1548,
    ):
        convolution_default = torch.ops.aten.convolution.default(
            primals_1543, primals_17, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1
        )
        add_tensor = torch.ops.aten.add.Tensor(primals_1544, 1)
        primals_1544 = None
        var_correction = torch.ops.aten.var.correction(
            convolution_default, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim = torch.ops.aten.mean.dim(convolution_default, [0, 2, 3], True)
        add_tensor_1 = torch.ops.aten.add.Tensor(var_correction, 0.001)
        var_correction = None
        sqrt_default = torch.ops.aten.sqrt.default(add_tensor_1)
        add_tensor_1 = None
        reciprocal_default = torch.ops.aten.reciprocal.default(sqrt_default)
        sqrt_default = None
        sub_tensor = torch.ops.aten.sub.Tensor(convolution_default, mean_dim)
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, reciprocal_default)
        sub_tensor = None
        squeeze_dim = torch.ops.aten.squeeze.dim(mean_dim, 3)
        mean_dim = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(squeeze_dim, 2)
        squeeze_dim = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(squeeze_dim_1, 0)
        squeeze_dim_1 = None
        squeeze_dim_3 = torch.ops.aten.squeeze.dim(reciprocal_default, 3)
        reciprocal_default = None
        squeeze_dim_4 = torch.ops.aten.squeeze.dim(squeeze_dim_3, 2)
        squeeze_dim_3 = None
        squeeze_dim_5 = torch.ops.aten.squeeze.dim(squeeze_dim_4, 0)
        squeeze_dim_4 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(primals_1547, -1)
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1)
        unsqueeze_default = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(primals_1548, -1)
        primals_1548 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1)
        unsqueeze_default_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1)
        mul_tensor = unsqueeze_default_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3)
        mul_tensor_6 = unsqueeze_default_3 = None
        relu_default = torch.ops.aten.relu.default(add_tensor_4)
        add_tensor_4 = None
        convolution_default_1 = torch.ops.aten.convolution.default(
            relu_default, primals_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_2 = torch.ops.aten.var.correction(
            convolution_default_1, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_1 = torch.ops.aten.mean.dim(convolution_default_1, [0, 2, 3], True)
        add_tensor_5 = torch.ops.aten.add.Tensor(var_correction_2, 0.001)
        var_correction_2 = None
        sqrt_default_1 = torch.ops.aten.sqrt.default(add_tensor_5)
        add_tensor_5 = None
        reciprocal_default_1 = torch.ops.aten.reciprocal.default(sqrt_default_1)
        sqrt_default_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convolution_default_1, mean_dim_1)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_1, reciprocal_default_1)
        sub_tensor_1 = None
        squeeze_dim_6 = torch.ops.aten.squeeze.dim(mean_dim_1, 3)
        mean_dim_1 = None
        squeeze_dim_7 = torch.ops.aten.squeeze.dim(squeeze_dim_6, 2)
        squeeze_dim_6 = None
        squeeze_dim_8 = torch.ops.aten.squeeze.dim(squeeze_dim_7, 0)
        squeeze_dim_7 = None
        squeeze_dim_9 = torch.ops.aten.squeeze.dim(reciprocal_default_1, 3)
        reciprocal_default_1 = None
        squeeze_dim_10 = torch.ops.aten.squeeze.dim(squeeze_dim_9, 2)
        squeeze_dim_9 = None
        squeeze_dim_11 = torch.ops.aten.squeeze.dim(squeeze_dim_10, 0)
        squeeze_dim_10 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1)
        unsqueeze_default_4 = None
        unsqueeze_default_6 = torch.ops.aten.unsqueeze.default(primals_20, -1)
        primals_20 = None
        unsqueeze_default_7 = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1)
        unsqueeze_default_6 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5)
        mul_tensor_7 = unsqueeze_default_5 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7)
        mul_tensor_13 = unsqueeze_default_7 = None
        relu_default_1 = torch.ops.aten.relu.default(add_tensor_8)
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(
            relu_default_1, [2, 2, 2, 2], 0.0
        )
        convolution_default_2 = torch.ops.aten.convolution.default(
            constant_pad_nd_default,
            primals_1,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            42,
        )
        convolution_default_3 = torch.ops.aten.convolution.default(
            convolution_default_2,
            primals_21,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_4 = torch.ops.aten.var.correction(
            convolution_default_3, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_2 = torch.ops.aten.mean.dim(convolution_default_3, [0, 2, 3], True)
        add_tensor_9 = torch.ops.aten.add.Tensor(var_correction_4, 0.001)
        var_correction_4 = None
        sqrt_default_2 = torch.ops.aten.sqrt.default(add_tensor_9)
        add_tensor_9 = None
        reciprocal_default_2 = torch.ops.aten.reciprocal.default(sqrt_default_2)
        sqrt_default_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convolution_default_3, mean_dim_2)
        mul_tensor_14 = torch.ops.aten.mul.Tensor(sub_tensor_2, reciprocal_default_2)
        sub_tensor_2 = None
        squeeze_dim_12 = torch.ops.aten.squeeze.dim(mean_dim_2, 3)
        mean_dim_2 = None
        squeeze_dim_13 = torch.ops.aten.squeeze.dim(squeeze_dim_12, 2)
        squeeze_dim_12 = None
        squeeze_dim_14 = torch.ops.aten.squeeze.dim(squeeze_dim_13, 0)
        squeeze_dim_13 = None
        squeeze_dim_15 = torch.ops.aten.squeeze.dim(reciprocal_default_2, 3)
        reciprocal_default_2 = None
        squeeze_dim_16 = torch.ops.aten.squeeze.dim(squeeze_dim_15, 2)
        squeeze_dim_15 = None
        squeeze_dim_17 = torch.ops.aten.squeeze.dim(squeeze_dim_16, 0)
        squeeze_dim_16 = None
        unsqueeze_default_8 = torch.ops.aten.unsqueeze.default(primals_22, -1)
        unsqueeze_default_9 = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1)
        unsqueeze_default_8 = None
        unsqueeze_default_10 = torch.ops.aten.unsqueeze.default(primals_23, -1)
        primals_23 = None
        unsqueeze_default_11 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_10, -1
        )
        unsqueeze_default_10 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(mul_tensor_14, unsqueeze_default_9)
        mul_tensor_14 = unsqueeze_default_9 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(mul_tensor_20, unsqueeze_default_11)
        mul_tensor_20 = unsqueeze_default_11 = None
        relu_default_2 = torch.ops.aten.relu.default(add_tensor_12)
        add_tensor_12 = None
        convolution_default_4 = torch.ops.aten.convolution.default(
            relu_default_2, primals_24, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 42
        )
        convolution_default_5 = torch.ops.aten.convolution.default(
            convolution_default_4,
            primals_25,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_6 = torch.ops.aten.var.correction(
            convolution_default_5, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_3 = torch.ops.aten.mean.dim(convolution_default_5, [0, 2, 3], True)
        add_tensor_13 = torch.ops.aten.add.Tensor(var_correction_6, 0.001)
        var_correction_6 = None
        sqrt_default_3 = torch.ops.aten.sqrt.default(add_tensor_13)
        add_tensor_13 = None
        reciprocal_default_3 = torch.ops.aten.reciprocal.default(sqrt_default_3)
        sqrt_default_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convolution_default_5, mean_dim_3)
        mul_tensor_21 = torch.ops.aten.mul.Tensor(sub_tensor_3, reciprocal_default_3)
        sub_tensor_3 = None
        squeeze_dim_18 = torch.ops.aten.squeeze.dim(mean_dim_3, 3)
        mean_dim_3 = None
        squeeze_dim_19 = torch.ops.aten.squeeze.dim(squeeze_dim_18, 2)
        squeeze_dim_18 = None
        squeeze_dim_20 = torch.ops.aten.squeeze.dim(squeeze_dim_19, 0)
        squeeze_dim_19 = None
        squeeze_dim_21 = torch.ops.aten.squeeze.dim(reciprocal_default_3, 3)
        reciprocal_default_3 = None
        squeeze_dim_22 = torch.ops.aten.squeeze.dim(squeeze_dim_21, 2)
        squeeze_dim_21 = None
        squeeze_dim_23 = torch.ops.aten.squeeze.dim(squeeze_dim_22, 0)
        squeeze_dim_22 = None
        unsqueeze_default_12 = torch.ops.aten.unsqueeze.default(primals_26, -1)
        unsqueeze_default_13 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_12, -1
        )
        unsqueeze_default_12 = None
        unsqueeze_default_14 = torch.ops.aten.unsqueeze.default(primals_27, -1)
        primals_27 = None
        unsqueeze_default_15 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_14, -1
        )
        unsqueeze_default_14 = None
        mul_tensor_27 = torch.ops.aten.mul.Tensor(mul_tensor_21, unsqueeze_default_13)
        mul_tensor_21 = unsqueeze_default_13 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(mul_tensor_27, unsqueeze_default_15)
        mul_tensor_27 = unsqueeze_default_15 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(
            relu_default, [3, 3, 3, 3], 0.0
        )
        convolution_default_6 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_1,
            primals_2,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            96,
        )
        convolution_default_7 = torch.ops.aten.convolution.default(
            convolution_default_6,
            primals_28,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_8 = torch.ops.aten.var.correction(
            convolution_default_7, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_4 = torch.ops.aten.mean.dim(convolution_default_7, [0, 2, 3], True)
        add_tensor_17 = torch.ops.aten.add.Tensor(var_correction_8, 0.001)
        var_correction_8 = None
        sqrt_default_4 = torch.ops.aten.sqrt.default(add_tensor_17)
        add_tensor_17 = None
        reciprocal_default_4 = torch.ops.aten.reciprocal.default(sqrt_default_4)
        sqrt_default_4 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convolution_default_7, mean_dim_4)
        mul_tensor_28 = torch.ops.aten.mul.Tensor(sub_tensor_4, reciprocal_default_4)
        sub_tensor_4 = None
        squeeze_dim_24 = torch.ops.aten.squeeze.dim(mean_dim_4, 3)
        mean_dim_4 = None
        squeeze_dim_25 = torch.ops.aten.squeeze.dim(squeeze_dim_24, 2)
        squeeze_dim_24 = None
        squeeze_dim_26 = torch.ops.aten.squeeze.dim(squeeze_dim_25, 0)
        squeeze_dim_25 = None
        squeeze_dim_27 = torch.ops.aten.squeeze.dim(reciprocal_default_4, 3)
        reciprocal_default_4 = None
        squeeze_dim_28 = torch.ops.aten.squeeze.dim(squeeze_dim_27, 2)
        squeeze_dim_27 = None
        squeeze_dim_29 = torch.ops.aten.squeeze.dim(squeeze_dim_28, 0)
        squeeze_dim_28 = None
        unsqueeze_default_16 = torch.ops.aten.unsqueeze.default(primals_29, -1)
        unsqueeze_default_17 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_16, -1
        )
        unsqueeze_default_16 = None
        unsqueeze_default_18 = torch.ops.aten.unsqueeze.default(primals_30, -1)
        primals_30 = None
        unsqueeze_default_19 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_18, -1
        )
        unsqueeze_default_18 = None
        mul_tensor_34 = torch.ops.aten.mul.Tensor(mul_tensor_28, unsqueeze_default_17)
        mul_tensor_28 = unsqueeze_default_17 = None
        add_tensor_20 = torch.ops.aten.add.Tensor(mul_tensor_34, unsqueeze_default_19)
        mul_tensor_34 = unsqueeze_default_19 = None
        relu_default_4 = torch.ops.aten.relu.default(add_tensor_20)
        add_tensor_20 = None
        convolution_default_8 = torch.ops.aten.convolution.default(
            relu_default_4, primals_31, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 42
        )
        convolution_default_9 = torch.ops.aten.convolution.default(
            convolution_default_8,
            primals_32,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_10 = torch.ops.aten.var.correction(
            convolution_default_9, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_5 = torch.ops.aten.mean.dim(convolution_default_9, [0, 2, 3], True)
        add_tensor_21 = torch.ops.aten.add.Tensor(var_correction_10, 0.001)
        var_correction_10 = None
        sqrt_default_5 = torch.ops.aten.sqrt.default(add_tensor_21)
        add_tensor_21 = None
        reciprocal_default_5 = torch.ops.aten.reciprocal.default(sqrt_default_5)
        sqrt_default_5 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convolution_default_9, mean_dim_5)
        mul_tensor_35 = torch.ops.aten.mul.Tensor(sub_tensor_5, reciprocal_default_5)
        sub_tensor_5 = None
        squeeze_dim_30 = torch.ops.aten.squeeze.dim(mean_dim_5, 3)
        mean_dim_5 = None
        squeeze_dim_31 = torch.ops.aten.squeeze.dim(squeeze_dim_30, 2)
        squeeze_dim_30 = None
        squeeze_dim_32 = torch.ops.aten.squeeze.dim(squeeze_dim_31, 0)
        squeeze_dim_31 = None
        squeeze_dim_33 = torch.ops.aten.squeeze.dim(reciprocal_default_5, 3)
        reciprocal_default_5 = None
        squeeze_dim_34 = torch.ops.aten.squeeze.dim(squeeze_dim_33, 2)
        squeeze_dim_33 = None
        squeeze_dim_35 = torch.ops.aten.squeeze.dim(squeeze_dim_34, 0)
        squeeze_dim_34 = None
        unsqueeze_default_20 = torch.ops.aten.unsqueeze.default(primals_33, -1)
        unsqueeze_default_21 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_20, -1
        )
        unsqueeze_default_20 = None
        unsqueeze_default_22 = torch.ops.aten.unsqueeze.default(primals_34, -1)
        primals_34 = None
        unsqueeze_default_23 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_22, -1
        )
        unsqueeze_default_22 = None
        mul_tensor_41 = torch.ops.aten.mul.Tensor(mul_tensor_35, unsqueeze_default_21)
        mul_tensor_35 = unsqueeze_default_21 = None
        add_tensor_24 = torch.ops.aten.add.Tensor(mul_tensor_41, unsqueeze_default_23)
        mul_tensor_41 = unsqueeze_default_23 = None
        add_tensor_25 = torch.ops.aten.add.Tensor(add_tensor_16, add_tensor_24)
        add_tensor_16 = add_tensor_24 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(
            add_tensor_8, [1, 1, 1, 1], -inf
        )
        max_pool2d_with_indices_default = (
            torch.ops.aten.max_pool2d_with_indices.default(
                constant_pad_nd_default_2, [3, 3], [2, 2]
            )
        )
        getitem = max_pool2d_with_indices_default[0]
        getitem_1 = max_pool2d_with_indices_default[1]
        max_pool2d_with_indices_default = None
        convolution_default_10 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_1,
            primals_3,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            96,
        )
        convolution_default_11 = torch.ops.aten.convolution.default(
            convolution_default_10,
            primals_35,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_12 = torch.ops.aten.var.correction(
            convolution_default_11, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_6 = torch.ops.aten.mean.dim(convolution_default_11, [0, 2, 3], True)
        add_tensor_26 = torch.ops.aten.add.Tensor(var_correction_12, 0.001)
        var_correction_12 = None
        sqrt_default_6 = torch.ops.aten.sqrt.default(add_tensor_26)
        add_tensor_26 = None
        reciprocal_default_6 = torch.ops.aten.reciprocal.default(sqrt_default_6)
        sqrt_default_6 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(convolution_default_11, mean_dim_6)
        mul_tensor_42 = torch.ops.aten.mul.Tensor(sub_tensor_6, reciprocal_default_6)
        sub_tensor_6 = None
        squeeze_dim_36 = torch.ops.aten.squeeze.dim(mean_dim_6, 3)
        mean_dim_6 = None
        squeeze_dim_37 = torch.ops.aten.squeeze.dim(squeeze_dim_36, 2)
        squeeze_dim_36 = None
        squeeze_dim_38 = torch.ops.aten.squeeze.dim(squeeze_dim_37, 0)
        squeeze_dim_37 = None
        squeeze_dim_39 = torch.ops.aten.squeeze.dim(reciprocal_default_6, 3)
        reciprocal_default_6 = None
        squeeze_dim_40 = torch.ops.aten.squeeze.dim(squeeze_dim_39, 2)
        squeeze_dim_39 = None
        squeeze_dim_41 = torch.ops.aten.squeeze.dim(squeeze_dim_40, 0)
        squeeze_dim_40 = None
        unsqueeze_default_24 = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_default_25 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_24, -1
        )
        unsqueeze_default_24 = None
        unsqueeze_default_26 = torch.ops.aten.unsqueeze.default(primals_37, -1)
        primals_37 = None
        unsqueeze_default_27 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_26, -1
        )
        unsqueeze_default_26 = None
        mul_tensor_48 = torch.ops.aten.mul.Tensor(mul_tensor_42, unsqueeze_default_25)
        mul_tensor_42 = unsqueeze_default_25 = None
        add_tensor_29 = torch.ops.aten.add.Tensor(mul_tensor_48, unsqueeze_default_27)
        mul_tensor_48 = unsqueeze_default_27 = None
        relu_default_6 = torch.ops.aten.relu.default(add_tensor_29)
        add_tensor_29 = None
        convolution_default_12 = torch.ops.aten.convolution.default(
            relu_default_6, primals_38, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 42
        )
        convolution_default_13 = torch.ops.aten.convolution.default(
            convolution_default_12,
            primals_39,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_14 = torch.ops.aten.var.correction(
            convolution_default_13, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_7 = torch.ops.aten.mean.dim(convolution_default_13, [0, 2, 3], True)
        add_tensor_30 = torch.ops.aten.add.Tensor(var_correction_14, 0.001)
        var_correction_14 = None
        sqrt_default_7 = torch.ops.aten.sqrt.default(add_tensor_30)
        add_tensor_30 = None
        reciprocal_default_7 = torch.ops.aten.reciprocal.default(sqrt_default_7)
        sqrt_default_7 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(convolution_default_13, mean_dim_7)
        mul_tensor_49 = torch.ops.aten.mul.Tensor(sub_tensor_7, reciprocal_default_7)
        sub_tensor_7 = None
        squeeze_dim_42 = torch.ops.aten.squeeze.dim(mean_dim_7, 3)
        mean_dim_7 = None
        squeeze_dim_43 = torch.ops.aten.squeeze.dim(squeeze_dim_42, 2)
        squeeze_dim_42 = None
        squeeze_dim_44 = torch.ops.aten.squeeze.dim(squeeze_dim_43, 0)
        squeeze_dim_43 = None
        squeeze_dim_45 = torch.ops.aten.squeeze.dim(reciprocal_default_7, 3)
        reciprocal_default_7 = None
        squeeze_dim_46 = torch.ops.aten.squeeze.dim(squeeze_dim_45, 2)
        squeeze_dim_45 = None
        squeeze_dim_47 = torch.ops.aten.squeeze.dim(squeeze_dim_46, 0)
        squeeze_dim_46 = None
        unsqueeze_default_28 = torch.ops.aten.unsqueeze.default(primals_40, -1)
        unsqueeze_default_29 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_28, -1
        )
        unsqueeze_default_28 = None
        unsqueeze_default_30 = torch.ops.aten.unsqueeze.default(primals_41, -1)
        primals_41 = None
        unsqueeze_default_31 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_30, -1
        )
        unsqueeze_default_30 = None
        mul_tensor_55 = torch.ops.aten.mul.Tensor(mul_tensor_49, unsqueeze_default_29)
        mul_tensor_49 = unsqueeze_default_29 = None
        add_tensor_33 = torch.ops.aten.add.Tensor(mul_tensor_55, unsqueeze_default_31)
        mul_tensor_55 = unsqueeze_default_31 = None
        add_tensor_34 = torch.ops.aten.add.Tensor(getitem, add_tensor_33)
        add_tensor_33 = None
        constant_pad_nd_default_4 = torch.ops.aten.constant_pad_nd.default(
            add_tensor_8, [1, 1, 1, 1], 0.0
        )
        add_tensor_8 = None
        avg_pool2d_default = torch.ops.aten.avg_pool2d.default(
            constant_pad_nd_default_4, [3, 3], [2, 2], [0, 0], False, False
        )
        constant_pad_nd_default_5 = torch.ops.aten.constant_pad_nd.default(
            relu_default, [2, 2, 2, 2], 0.0
        )
        convolution_default_14 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_5,
            primals_4,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            96,
        )
        convolution_default_15 = torch.ops.aten.convolution.default(
            convolution_default_14,
            primals_42,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_16 = torch.ops.aten.var.correction(
            convolution_default_15, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_8 = torch.ops.aten.mean.dim(convolution_default_15, [0, 2, 3], True)
        add_tensor_35 = torch.ops.aten.add.Tensor(var_correction_16, 0.001)
        var_correction_16 = None
        sqrt_default_8 = torch.ops.aten.sqrt.default(add_tensor_35)
        add_tensor_35 = None
        reciprocal_default_8 = torch.ops.aten.reciprocal.default(sqrt_default_8)
        sqrt_default_8 = None
        sub_tensor_8 = torch.ops.aten.sub.Tensor(convolution_default_15, mean_dim_8)
        mul_tensor_56 = torch.ops.aten.mul.Tensor(sub_tensor_8, reciprocal_default_8)
        sub_tensor_8 = None
        squeeze_dim_48 = torch.ops.aten.squeeze.dim(mean_dim_8, 3)
        mean_dim_8 = None
        squeeze_dim_49 = torch.ops.aten.squeeze.dim(squeeze_dim_48, 2)
        squeeze_dim_48 = None
        squeeze_dim_50 = torch.ops.aten.squeeze.dim(squeeze_dim_49, 0)
        squeeze_dim_49 = None
        squeeze_dim_51 = torch.ops.aten.squeeze.dim(reciprocal_default_8, 3)
        reciprocal_default_8 = None
        squeeze_dim_52 = torch.ops.aten.squeeze.dim(squeeze_dim_51, 2)
        squeeze_dim_51 = None
        squeeze_dim_53 = torch.ops.aten.squeeze.dim(squeeze_dim_52, 0)
        squeeze_dim_52 = None
        unsqueeze_default_32 = torch.ops.aten.unsqueeze.default(primals_43, -1)
        unsqueeze_default_33 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_32, -1
        )
        unsqueeze_default_32 = None
        unsqueeze_default_34 = torch.ops.aten.unsqueeze.default(primals_44, -1)
        primals_44 = None
        unsqueeze_default_35 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_34, -1
        )
        unsqueeze_default_34 = None
        mul_tensor_62 = torch.ops.aten.mul.Tensor(mul_tensor_56, unsqueeze_default_33)
        mul_tensor_56 = unsqueeze_default_33 = None
        add_tensor_38 = torch.ops.aten.add.Tensor(mul_tensor_62, unsqueeze_default_35)
        mul_tensor_62 = unsqueeze_default_35 = None
        relu_default_8 = torch.ops.aten.relu.default(add_tensor_38)
        add_tensor_38 = None
        convolution_default_16 = torch.ops.aten.convolution.default(
            relu_default_8, primals_45, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 42
        )
        convolution_default_17 = torch.ops.aten.convolution.default(
            convolution_default_16,
            primals_46,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_18 = torch.ops.aten.var.correction(
            convolution_default_17, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_9 = torch.ops.aten.mean.dim(convolution_default_17, [0, 2, 3], True)
        add_tensor_39 = torch.ops.aten.add.Tensor(var_correction_18, 0.001)
        var_correction_18 = None
        sqrt_default_9 = torch.ops.aten.sqrt.default(add_tensor_39)
        add_tensor_39 = None
        reciprocal_default_9 = torch.ops.aten.reciprocal.default(sqrt_default_9)
        sqrt_default_9 = None
        sub_tensor_9 = torch.ops.aten.sub.Tensor(convolution_default_17, mean_dim_9)
        mul_tensor_63 = torch.ops.aten.mul.Tensor(sub_tensor_9, reciprocal_default_9)
        sub_tensor_9 = None
        squeeze_dim_54 = torch.ops.aten.squeeze.dim(mean_dim_9, 3)
        mean_dim_9 = None
        squeeze_dim_55 = torch.ops.aten.squeeze.dim(squeeze_dim_54, 2)
        squeeze_dim_54 = None
        squeeze_dim_56 = torch.ops.aten.squeeze.dim(squeeze_dim_55, 0)
        squeeze_dim_55 = None
        squeeze_dim_57 = torch.ops.aten.squeeze.dim(reciprocal_default_9, 3)
        reciprocal_default_9 = None
        squeeze_dim_58 = torch.ops.aten.squeeze.dim(squeeze_dim_57, 2)
        squeeze_dim_57 = None
        squeeze_dim_59 = torch.ops.aten.squeeze.dim(squeeze_dim_58, 0)
        squeeze_dim_58 = None
        unsqueeze_default_36 = torch.ops.aten.unsqueeze.default(primals_47, -1)
        unsqueeze_default_37 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_36, -1
        )
        unsqueeze_default_36 = None
        unsqueeze_default_38 = torch.ops.aten.unsqueeze.default(primals_48, -1)
        primals_48 = None
        unsqueeze_default_39 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_38, -1
        )
        unsqueeze_default_38 = None
        mul_tensor_69 = torch.ops.aten.mul.Tensor(mul_tensor_63, unsqueeze_default_37)
        mul_tensor_63 = unsqueeze_default_37 = None
        add_tensor_42 = torch.ops.aten.add.Tensor(mul_tensor_69, unsqueeze_default_39)
        mul_tensor_69 = unsqueeze_default_39 = None
        add_tensor_43 = torch.ops.aten.add.Tensor(avg_pool2d_default, add_tensor_42)
        avg_pool2d_default = add_tensor_42 = None
        avg_pool2d_default_1 = torch.ops.aten.avg_pool2d.default(
            add_tensor_25, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_44 = torch.ops.aten.add.Tensor(avg_pool2d_default_1, add_tensor_34)
        avg_pool2d_default_1 = None
        relu_default_9 = torch.ops.aten.relu.default(add_tensor_25)
        convolution_default_18 = torch.ops.aten.convolution.default(
            relu_default_9, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 42
        )
        relu_default_9 = None
        convolution_default_19 = torch.ops.aten.convolution.default(
            convolution_default_18,
            primals_50,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_20 = torch.ops.aten.var.correction(
            convolution_default_19, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_10 = torch.ops.aten.mean.dim(convolution_default_19, [0, 2, 3], True)
        add_tensor_45 = torch.ops.aten.add.Tensor(var_correction_20, 0.001)
        var_correction_20 = None
        sqrt_default_10 = torch.ops.aten.sqrt.default(add_tensor_45)
        add_tensor_45 = None
        reciprocal_default_10 = torch.ops.aten.reciprocal.default(sqrt_default_10)
        sqrt_default_10 = None
        sub_tensor_10 = torch.ops.aten.sub.Tensor(convolution_default_19, mean_dim_10)
        mul_tensor_70 = torch.ops.aten.mul.Tensor(sub_tensor_10, reciprocal_default_10)
        sub_tensor_10 = None
        squeeze_dim_60 = torch.ops.aten.squeeze.dim(mean_dim_10, 3)
        mean_dim_10 = None
        squeeze_dim_61 = torch.ops.aten.squeeze.dim(squeeze_dim_60, 2)
        squeeze_dim_60 = None
        squeeze_dim_62 = torch.ops.aten.squeeze.dim(squeeze_dim_61, 0)
        squeeze_dim_61 = None
        squeeze_dim_63 = torch.ops.aten.squeeze.dim(reciprocal_default_10, 3)
        reciprocal_default_10 = None
        squeeze_dim_64 = torch.ops.aten.squeeze.dim(squeeze_dim_63, 2)
        squeeze_dim_63 = None
        squeeze_dim_65 = torch.ops.aten.squeeze.dim(squeeze_dim_64, 0)
        squeeze_dim_64 = None
        unsqueeze_default_40 = torch.ops.aten.unsqueeze.default(primals_51, -1)
        unsqueeze_default_41 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_40, -1
        )
        unsqueeze_default_40 = None
        unsqueeze_default_42 = torch.ops.aten.unsqueeze.default(primals_52, -1)
        primals_52 = None
        unsqueeze_default_43 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_42, -1
        )
        unsqueeze_default_42 = None
        mul_tensor_76 = torch.ops.aten.mul.Tensor(mul_tensor_70, unsqueeze_default_41)
        mul_tensor_70 = unsqueeze_default_41 = None
        add_tensor_48 = torch.ops.aten.add.Tensor(mul_tensor_76, unsqueeze_default_43)
        mul_tensor_76 = unsqueeze_default_43 = None
        relu_default_10 = torch.ops.aten.relu.default(add_tensor_48)
        add_tensor_48 = None
        convolution_default_20 = torch.ops.aten.convolution.default(
            relu_default_10, primals_53, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 42
        )
        convolution_default_21 = torch.ops.aten.convolution.default(
            convolution_default_20,
            primals_54,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_22 = torch.ops.aten.var.correction(
            convolution_default_21, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_11 = torch.ops.aten.mean.dim(convolution_default_21, [0, 2, 3], True)
        add_tensor_49 = torch.ops.aten.add.Tensor(var_correction_22, 0.001)
        var_correction_22 = None
        sqrt_default_11 = torch.ops.aten.sqrt.default(add_tensor_49)
        add_tensor_49 = None
        reciprocal_default_11 = torch.ops.aten.reciprocal.default(sqrt_default_11)
        sqrt_default_11 = None
        sub_tensor_11 = torch.ops.aten.sub.Tensor(convolution_default_21, mean_dim_11)
        mul_tensor_77 = torch.ops.aten.mul.Tensor(sub_tensor_11, reciprocal_default_11)
        sub_tensor_11 = None
        squeeze_dim_66 = torch.ops.aten.squeeze.dim(mean_dim_11, 3)
        mean_dim_11 = None
        squeeze_dim_67 = torch.ops.aten.squeeze.dim(squeeze_dim_66, 2)
        squeeze_dim_66 = None
        squeeze_dim_68 = torch.ops.aten.squeeze.dim(squeeze_dim_67, 0)
        squeeze_dim_67 = None
        squeeze_dim_69 = torch.ops.aten.squeeze.dim(reciprocal_default_11, 3)
        reciprocal_default_11 = None
        squeeze_dim_70 = torch.ops.aten.squeeze.dim(squeeze_dim_69, 2)
        squeeze_dim_69 = None
        squeeze_dim_71 = torch.ops.aten.squeeze.dim(squeeze_dim_70, 0)
        squeeze_dim_70 = None
        unsqueeze_default_44 = torch.ops.aten.unsqueeze.default(primals_55, -1)
        unsqueeze_default_45 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_44, -1
        )
        unsqueeze_default_44 = None
        unsqueeze_default_46 = torch.ops.aten.unsqueeze.default(primals_56, -1)
        primals_56 = None
        unsqueeze_default_47 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_46, -1
        )
        unsqueeze_default_46 = None
        mul_tensor_83 = torch.ops.aten.mul.Tensor(mul_tensor_77, unsqueeze_default_45)
        mul_tensor_77 = unsqueeze_default_45 = None
        add_tensor_52 = torch.ops.aten.add.Tensor(mul_tensor_83, unsqueeze_default_47)
        mul_tensor_83 = unsqueeze_default_47 = None
        add_tensor_53 = torch.ops.aten.add.Tensor(add_tensor_52, getitem)
        add_tensor_52 = getitem = None
        cat_default = torch.ops.aten.cat.default(
            [add_tensor_34, add_tensor_43, add_tensor_44, add_tensor_53], 1
        )
        add_tensor_34 = add_tensor_43 = add_tensor_44 = add_tensor_53 = None
        relu_default_11 = torch.ops.aten.relu.default(cat_default)
        cat_default = None
        convolution_default_22 = torch.ops.aten.convolution.default(
            relu_default_11, primals_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_24 = torch.ops.aten.var.correction(
            convolution_default_22, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_12 = torch.ops.aten.mean.dim(convolution_default_22, [0, 2, 3], True)
        add_tensor_54 = torch.ops.aten.add.Tensor(var_correction_24, 0.001)
        var_correction_24 = None
        sqrt_default_12 = torch.ops.aten.sqrt.default(add_tensor_54)
        add_tensor_54 = None
        reciprocal_default_12 = torch.ops.aten.reciprocal.default(sqrt_default_12)
        sqrt_default_12 = None
        sub_tensor_12 = torch.ops.aten.sub.Tensor(convolution_default_22, mean_dim_12)
        mul_tensor_84 = torch.ops.aten.mul.Tensor(sub_tensor_12, reciprocal_default_12)
        sub_tensor_12 = None
        squeeze_dim_72 = torch.ops.aten.squeeze.dim(mean_dim_12, 3)
        mean_dim_12 = None
        squeeze_dim_73 = torch.ops.aten.squeeze.dim(squeeze_dim_72, 2)
        squeeze_dim_72 = None
        squeeze_dim_74 = torch.ops.aten.squeeze.dim(squeeze_dim_73, 0)
        squeeze_dim_73 = None
        squeeze_dim_75 = torch.ops.aten.squeeze.dim(reciprocal_default_12, 3)
        reciprocal_default_12 = None
        squeeze_dim_76 = torch.ops.aten.squeeze.dim(squeeze_dim_75, 2)
        squeeze_dim_75 = None
        squeeze_dim_77 = torch.ops.aten.squeeze.dim(squeeze_dim_76, 0)
        squeeze_dim_76 = None
        unsqueeze_default_48 = torch.ops.aten.unsqueeze.default(primals_58, -1)
        unsqueeze_default_49 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_48, -1
        )
        unsqueeze_default_48 = None
        unsqueeze_default_50 = torch.ops.aten.unsqueeze.default(primals_59, -1)
        primals_59 = None
        unsqueeze_default_51 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_50, -1
        )
        unsqueeze_default_50 = None
        mul_tensor_90 = torch.ops.aten.mul.Tensor(mul_tensor_84, unsqueeze_default_49)
        mul_tensor_84 = unsqueeze_default_49 = None
        add_tensor_57 = torch.ops.aten.add.Tensor(mul_tensor_90, unsqueeze_default_51)
        mul_tensor_90 = unsqueeze_default_51 = None
        avg_pool2d_default_2 = torch.ops.aten.avg_pool2d.default(
            relu_default, [1, 1], [2, 2], [0, 0], False, False
        )
        convolution_default_23 = torch.ops.aten.convolution.default(
            avg_pool2d_default_2,
            primals_60,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        constant_pad_nd_default_7 = torch.ops.aten.constant_pad_nd.default(
            relu_default, [-1, 1, -1, 1], 0.0
        )
        avg_pool2d_default_3 = torch.ops.aten.avg_pool2d.default(
            constant_pad_nd_default_7, [1, 1], [2, 2], [0, 0], False, False
        )
        convolution_default_24 = torch.ops.aten.convolution.default(
            avg_pool2d_default_3,
            primals_61,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        cat_default_1 = torch.ops.aten.cat.default(
            [convolution_default_23, convolution_default_24], 1
        )
        convolution_default_23 = convolution_default_24 = None
        var_correction_26 = torch.ops.aten.var.correction(
            cat_default_1, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_13 = torch.ops.aten.mean.dim(cat_default_1, [0, 2, 3], True)
        add_tensor_58 = torch.ops.aten.add.Tensor(var_correction_26, 0.001)
        var_correction_26 = None
        sqrt_default_13 = torch.ops.aten.sqrt.default(add_tensor_58)
        add_tensor_58 = None
        reciprocal_default_13 = torch.ops.aten.reciprocal.default(sqrt_default_13)
        sqrt_default_13 = None
        sub_tensor_13 = torch.ops.aten.sub.Tensor(cat_default_1, mean_dim_13)
        mul_tensor_91 = torch.ops.aten.mul.Tensor(sub_tensor_13, reciprocal_default_13)
        sub_tensor_13 = None
        squeeze_dim_78 = torch.ops.aten.squeeze.dim(mean_dim_13, 3)
        mean_dim_13 = None
        squeeze_dim_79 = torch.ops.aten.squeeze.dim(squeeze_dim_78, 2)
        squeeze_dim_78 = None
        squeeze_dim_80 = torch.ops.aten.squeeze.dim(squeeze_dim_79, 0)
        squeeze_dim_79 = None
        squeeze_dim_81 = torch.ops.aten.squeeze.dim(reciprocal_default_13, 3)
        reciprocal_default_13 = None
        squeeze_dim_82 = torch.ops.aten.squeeze.dim(squeeze_dim_81, 2)
        squeeze_dim_81 = None
        squeeze_dim_83 = torch.ops.aten.squeeze.dim(squeeze_dim_82, 0)
        squeeze_dim_82 = None
        unsqueeze_default_52 = torch.ops.aten.unsqueeze.default(primals_62, -1)
        unsqueeze_default_53 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_52, -1
        )
        unsqueeze_default_52 = None
        unsqueeze_default_54 = torch.ops.aten.unsqueeze.default(primals_63, -1)
        primals_63 = None
        unsqueeze_default_55 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_54, -1
        )
        unsqueeze_default_54 = None
        mul_tensor_97 = torch.ops.aten.mul.Tensor(mul_tensor_91, unsqueeze_default_53)
        mul_tensor_91 = unsqueeze_default_53 = None
        add_tensor_61 = torch.ops.aten.add.Tensor(mul_tensor_97, unsqueeze_default_55)
        mul_tensor_97 = unsqueeze_default_55 = None
        relu_default_13 = torch.ops.aten.relu.default(add_tensor_57)
        constant_pad_nd_default_8 = torch.ops.aten.constant_pad_nd.default(
            relu_default_13, [2, 2, 2, 2], 0.0
        )
        convolution_default_25 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_8,
            primals_5,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            84,
        )
        convolution_default_26 = torch.ops.aten.convolution.default(
            convolution_default_25,
            primals_64,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_28 = torch.ops.aten.var.correction(
            convolution_default_26, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_14 = torch.ops.aten.mean.dim(convolution_default_26, [0, 2, 3], True)
        add_tensor_62 = torch.ops.aten.add.Tensor(var_correction_28, 0.001)
        var_correction_28 = None
        sqrt_default_14 = torch.ops.aten.sqrt.default(add_tensor_62)
        add_tensor_62 = None
        reciprocal_default_14 = torch.ops.aten.reciprocal.default(sqrt_default_14)
        sqrt_default_14 = None
        sub_tensor_14 = torch.ops.aten.sub.Tensor(convolution_default_26, mean_dim_14)
        mul_tensor_98 = torch.ops.aten.mul.Tensor(sub_tensor_14, reciprocal_default_14)
        sub_tensor_14 = None
        squeeze_dim_84 = torch.ops.aten.squeeze.dim(mean_dim_14, 3)
        mean_dim_14 = None
        squeeze_dim_85 = torch.ops.aten.squeeze.dim(squeeze_dim_84, 2)
        squeeze_dim_84 = None
        squeeze_dim_86 = torch.ops.aten.squeeze.dim(squeeze_dim_85, 0)
        squeeze_dim_85 = None
        squeeze_dim_87 = torch.ops.aten.squeeze.dim(reciprocal_default_14, 3)
        reciprocal_default_14 = None
        squeeze_dim_88 = torch.ops.aten.squeeze.dim(squeeze_dim_87, 2)
        squeeze_dim_87 = None
        squeeze_dim_89 = torch.ops.aten.squeeze.dim(squeeze_dim_88, 0)
        squeeze_dim_88 = None
        unsqueeze_default_56 = torch.ops.aten.unsqueeze.default(primals_65, -1)
        unsqueeze_default_57 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_56, -1
        )
        unsqueeze_default_56 = None
        unsqueeze_default_58 = torch.ops.aten.unsqueeze.default(primals_66, -1)
        primals_66 = None
        unsqueeze_default_59 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_58, -1
        )
        unsqueeze_default_58 = None
        mul_tensor_104 = torch.ops.aten.mul.Tensor(mul_tensor_98, unsqueeze_default_57)
        mul_tensor_98 = unsqueeze_default_57 = None
        add_tensor_65 = torch.ops.aten.add.Tensor(mul_tensor_104, unsqueeze_default_59)
        mul_tensor_104 = unsqueeze_default_59 = None
        relu_default_14 = torch.ops.aten.relu.default(add_tensor_65)
        add_tensor_65 = None
        convolution_default_27 = torch.ops.aten.convolution.default(
            relu_default_14, primals_67, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 84
        )
        convolution_default_28 = torch.ops.aten.convolution.default(
            convolution_default_27,
            primals_68,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_30 = torch.ops.aten.var.correction(
            convolution_default_28, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_15 = torch.ops.aten.mean.dim(convolution_default_28, [0, 2, 3], True)
        add_tensor_66 = torch.ops.aten.add.Tensor(var_correction_30, 0.001)
        var_correction_30 = None
        sqrt_default_15 = torch.ops.aten.sqrt.default(add_tensor_66)
        add_tensor_66 = None
        reciprocal_default_15 = torch.ops.aten.reciprocal.default(sqrt_default_15)
        sqrt_default_15 = None
        sub_tensor_15 = torch.ops.aten.sub.Tensor(convolution_default_28, mean_dim_15)
        mul_tensor_105 = torch.ops.aten.mul.Tensor(sub_tensor_15, reciprocal_default_15)
        sub_tensor_15 = None
        squeeze_dim_90 = torch.ops.aten.squeeze.dim(mean_dim_15, 3)
        mean_dim_15 = None
        squeeze_dim_91 = torch.ops.aten.squeeze.dim(squeeze_dim_90, 2)
        squeeze_dim_90 = None
        squeeze_dim_92 = torch.ops.aten.squeeze.dim(squeeze_dim_91, 0)
        squeeze_dim_91 = None
        squeeze_dim_93 = torch.ops.aten.squeeze.dim(reciprocal_default_15, 3)
        reciprocal_default_15 = None
        squeeze_dim_94 = torch.ops.aten.squeeze.dim(squeeze_dim_93, 2)
        squeeze_dim_93 = None
        squeeze_dim_95 = torch.ops.aten.squeeze.dim(squeeze_dim_94, 0)
        squeeze_dim_94 = None
        unsqueeze_default_60 = torch.ops.aten.unsqueeze.default(primals_69, -1)
        unsqueeze_default_61 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_60, -1
        )
        unsqueeze_default_60 = None
        unsqueeze_default_62 = torch.ops.aten.unsqueeze.default(primals_70, -1)
        primals_70 = None
        unsqueeze_default_63 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_62, -1
        )
        unsqueeze_default_62 = None
        mul_tensor_111 = torch.ops.aten.mul.Tensor(mul_tensor_105, unsqueeze_default_61)
        mul_tensor_105 = unsqueeze_default_61 = None
        add_tensor_69 = torch.ops.aten.add.Tensor(mul_tensor_111, unsqueeze_default_63)
        mul_tensor_111 = unsqueeze_default_63 = None
        relu_default_15 = torch.ops.aten.relu.default(add_tensor_61)
        add_tensor_61 = None
        constant_pad_nd_default_9 = torch.ops.aten.constant_pad_nd.default(
            relu_default_15, [3, 3, 3, 3], 0.0
        )
        convolution_default_29 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_9,
            primals_6,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            84,
        )
        convolution_default_30 = torch.ops.aten.convolution.default(
            convolution_default_29,
            primals_71,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_32 = torch.ops.aten.var.correction(
            convolution_default_30, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_16 = torch.ops.aten.mean.dim(convolution_default_30, [0, 2, 3], True)
        add_tensor_70 = torch.ops.aten.add.Tensor(var_correction_32, 0.001)
        var_correction_32 = None
        sqrt_default_16 = torch.ops.aten.sqrt.default(add_tensor_70)
        add_tensor_70 = None
        reciprocal_default_16 = torch.ops.aten.reciprocal.default(sqrt_default_16)
        sqrt_default_16 = None
        sub_tensor_16 = torch.ops.aten.sub.Tensor(convolution_default_30, mean_dim_16)
        mul_tensor_112 = torch.ops.aten.mul.Tensor(sub_tensor_16, reciprocal_default_16)
        sub_tensor_16 = None
        squeeze_dim_96 = torch.ops.aten.squeeze.dim(mean_dim_16, 3)
        mean_dim_16 = None
        squeeze_dim_97 = torch.ops.aten.squeeze.dim(squeeze_dim_96, 2)
        squeeze_dim_96 = None
        squeeze_dim_98 = torch.ops.aten.squeeze.dim(squeeze_dim_97, 0)
        squeeze_dim_97 = None
        squeeze_dim_99 = torch.ops.aten.squeeze.dim(reciprocal_default_16, 3)
        reciprocal_default_16 = None
        squeeze_dim_100 = torch.ops.aten.squeeze.dim(squeeze_dim_99, 2)
        squeeze_dim_99 = None
        squeeze_dim_101 = torch.ops.aten.squeeze.dim(squeeze_dim_100, 0)
        squeeze_dim_100 = None
        unsqueeze_default_64 = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_default_65 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_64, -1
        )
        unsqueeze_default_64 = None
        unsqueeze_default_66 = torch.ops.aten.unsqueeze.default(primals_73, -1)
        primals_73 = None
        unsqueeze_default_67 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_66, -1
        )
        unsqueeze_default_66 = None
        mul_tensor_118 = torch.ops.aten.mul.Tensor(mul_tensor_112, unsqueeze_default_65)
        mul_tensor_112 = unsqueeze_default_65 = None
        add_tensor_73 = torch.ops.aten.add.Tensor(mul_tensor_118, unsqueeze_default_67)
        mul_tensor_118 = unsqueeze_default_67 = None
        relu_default_16 = torch.ops.aten.relu.default(add_tensor_73)
        add_tensor_73 = None
        convolution_default_31 = torch.ops.aten.convolution.default(
            relu_default_16, primals_74, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 84
        )
        convolution_default_32 = torch.ops.aten.convolution.default(
            convolution_default_31,
            primals_75,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_34 = torch.ops.aten.var.correction(
            convolution_default_32, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_17 = torch.ops.aten.mean.dim(convolution_default_32, [0, 2, 3], True)
        add_tensor_74 = torch.ops.aten.add.Tensor(var_correction_34, 0.001)
        var_correction_34 = None
        sqrt_default_17 = torch.ops.aten.sqrt.default(add_tensor_74)
        add_tensor_74 = None
        reciprocal_default_17 = torch.ops.aten.reciprocal.default(sqrt_default_17)
        sqrt_default_17 = None
        sub_tensor_17 = torch.ops.aten.sub.Tensor(convolution_default_32, mean_dim_17)
        mul_tensor_119 = torch.ops.aten.mul.Tensor(sub_tensor_17, reciprocal_default_17)
        sub_tensor_17 = None
        squeeze_dim_102 = torch.ops.aten.squeeze.dim(mean_dim_17, 3)
        mean_dim_17 = None
        squeeze_dim_103 = torch.ops.aten.squeeze.dim(squeeze_dim_102, 2)
        squeeze_dim_102 = None
        squeeze_dim_104 = torch.ops.aten.squeeze.dim(squeeze_dim_103, 0)
        squeeze_dim_103 = None
        squeeze_dim_105 = torch.ops.aten.squeeze.dim(reciprocal_default_17, 3)
        reciprocal_default_17 = None
        squeeze_dim_106 = torch.ops.aten.squeeze.dim(squeeze_dim_105, 2)
        squeeze_dim_105 = None
        squeeze_dim_107 = torch.ops.aten.squeeze.dim(squeeze_dim_106, 0)
        squeeze_dim_106 = None
        unsqueeze_default_68 = torch.ops.aten.unsqueeze.default(primals_76, -1)
        unsqueeze_default_69 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_68, -1
        )
        unsqueeze_default_68 = None
        unsqueeze_default_70 = torch.ops.aten.unsqueeze.default(primals_77, -1)
        primals_77 = None
        unsqueeze_default_71 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_70, -1
        )
        unsqueeze_default_70 = None
        mul_tensor_125 = torch.ops.aten.mul.Tensor(mul_tensor_119, unsqueeze_default_69)
        mul_tensor_119 = unsqueeze_default_69 = None
        add_tensor_77 = torch.ops.aten.add.Tensor(mul_tensor_125, unsqueeze_default_71)
        mul_tensor_125 = unsqueeze_default_71 = None
        add_tensor_78 = torch.ops.aten.add.Tensor(add_tensor_69, add_tensor_77)
        add_tensor_69 = add_tensor_77 = None
        constant_pad_nd_default_10 = torch.ops.aten.constant_pad_nd.default(
            add_tensor_57, [1, 1, 1, 1], -inf
        )
        max_pool2d_with_indices_default_2 = (
            torch.ops.aten.max_pool2d_with_indices.default(
                constant_pad_nd_default_10, [3, 3], [2, 2]
            )
        )
        getitem_4 = max_pool2d_with_indices_default_2[0]
        getitem_5 = max_pool2d_with_indices_default_2[1]
        max_pool2d_with_indices_default_2 = None
        convolution_default_33 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_9,
            primals_7,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            84,
        )
        convolution_default_34 = torch.ops.aten.convolution.default(
            convolution_default_33,
            primals_78,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_36 = torch.ops.aten.var.correction(
            convolution_default_34, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_18 = torch.ops.aten.mean.dim(convolution_default_34, [0, 2, 3], True)
        add_tensor_79 = torch.ops.aten.add.Tensor(var_correction_36, 0.001)
        var_correction_36 = None
        sqrt_default_18 = torch.ops.aten.sqrt.default(add_tensor_79)
        add_tensor_79 = None
        reciprocal_default_18 = torch.ops.aten.reciprocal.default(sqrt_default_18)
        sqrt_default_18 = None
        sub_tensor_18 = torch.ops.aten.sub.Tensor(convolution_default_34, mean_dim_18)
        mul_tensor_126 = torch.ops.aten.mul.Tensor(sub_tensor_18, reciprocal_default_18)
        sub_tensor_18 = None
        squeeze_dim_108 = torch.ops.aten.squeeze.dim(mean_dim_18, 3)
        mean_dim_18 = None
        squeeze_dim_109 = torch.ops.aten.squeeze.dim(squeeze_dim_108, 2)
        squeeze_dim_108 = None
        squeeze_dim_110 = torch.ops.aten.squeeze.dim(squeeze_dim_109, 0)
        squeeze_dim_109 = None
        squeeze_dim_111 = torch.ops.aten.squeeze.dim(reciprocal_default_18, 3)
        reciprocal_default_18 = None
        squeeze_dim_112 = torch.ops.aten.squeeze.dim(squeeze_dim_111, 2)
        squeeze_dim_111 = None
        squeeze_dim_113 = torch.ops.aten.squeeze.dim(squeeze_dim_112, 0)
        squeeze_dim_112 = None
        unsqueeze_default_72 = torch.ops.aten.unsqueeze.default(primals_79, -1)
        unsqueeze_default_73 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_72, -1
        )
        unsqueeze_default_72 = None
        unsqueeze_default_74 = torch.ops.aten.unsqueeze.default(primals_80, -1)
        primals_80 = None
        unsqueeze_default_75 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_74, -1
        )
        unsqueeze_default_74 = None
        mul_tensor_132 = torch.ops.aten.mul.Tensor(mul_tensor_126, unsqueeze_default_73)
        mul_tensor_126 = unsqueeze_default_73 = None
        add_tensor_82 = torch.ops.aten.add.Tensor(mul_tensor_132, unsqueeze_default_75)
        mul_tensor_132 = unsqueeze_default_75 = None
        relu_default_18 = torch.ops.aten.relu.default(add_tensor_82)
        add_tensor_82 = None
        convolution_default_35 = torch.ops.aten.convolution.default(
            relu_default_18, primals_81, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 84
        )
        convolution_default_36 = torch.ops.aten.convolution.default(
            convolution_default_35,
            primals_82,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_38 = torch.ops.aten.var.correction(
            convolution_default_36, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_19 = torch.ops.aten.mean.dim(convolution_default_36, [0, 2, 3], True)
        add_tensor_83 = torch.ops.aten.add.Tensor(var_correction_38, 0.001)
        var_correction_38 = None
        sqrt_default_19 = torch.ops.aten.sqrt.default(add_tensor_83)
        add_tensor_83 = None
        reciprocal_default_19 = torch.ops.aten.reciprocal.default(sqrt_default_19)
        sqrt_default_19 = None
        sub_tensor_19 = torch.ops.aten.sub.Tensor(convolution_default_36, mean_dim_19)
        mul_tensor_133 = torch.ops.aten.mul.Tensor(sub_tensor_19, reciprocal_default_19)
        sub_tensor_19 = None
        squeeze_dim_114 = torch.ops.aten.squeeze.dim(mean_dim_19, 3)
        mean_dim_19 = None
        squeeze_dim_115 = torch.ops.aten.squeeze.dim(squeeze_dim_114, 2)
        squeeze_dim_114 = None
        squeeze_dim_116 = torch.ops.aten.squeeze.dim(squeeze_dim_115, 0)
        squeeze_dim_115 = None
        squeeze_dim_117 = torch.ops.aten.squeeze.dim(reciprocal_default_19, 3)
        reciprocal_default_19 = None
        squeeze_dim_118 = torch.ops.aten.squeeze.dim(squeeze_dim_117, 2)
        squeeze_dim_117 = None
        squeeze_dim_119 = torch.ops.aten.squeeze.dim(squeeze_dim_118, 0)
        squeeze_dim_118 = None
        unsqueeze_default_76 = torch.ops.aten.unsqueeze.default(primals_83, -1)
        unsqueeze_default_77 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_76, -1
        )
        unsqueeze_default_76 = None
        unsqueeze_default_78 = torch.ops.aten.unsqueeze.default(primals_84, -1)
        primals_84 = None
        unsqueeze_default_79 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_78, -1
        )
        unsqueeze_default_78 = None
        mul_tensor_139 = torch.ops.aten.mul.Tensor(mul_tensor_133, unsqueeze_default_77)
        mul_tensor_133 = unsqueeze_default_77 = None
        add_tensor_86 = torch.ops.aten.add.Tensor(mul_tensor_139, unsqueeze_default_79)
        mul_tensor_139 = unsqueeze_default_79 = None
        add_tensor_87 = torch.ops.aten.add.Tensor(getitem_4, add_tensor_86)
        add_tensor_86 = None
        constant_pad_nd_default_12 = torch.ops.aten.constant_pad_nd.default(
            add_tensor_57, [1, 1, 1, 1], 0.0
        )
        add_tensor_57 = None
        avg_pool2d_default_4 = torch.ops.aten.avg_pool2d.default(
            constant_pad_nd_default_12, [3, 3], [2, 2], [0, 0], False, False
        )
        constant_pad_nd_default_13 = torch.ops.aten.constant_pad_nd.default(
            relu_default_15, [2, 2, 2, 2], 0.0
        )
        convolution_default_37 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_13,
            primals_8,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            84,
        )
        convolution_default_38 = torch.ops.aten.convolution.default(
            convolution_default_37,
            primals_85,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_40 = torch.ops.aten.var.correction(
            convolution_default_38, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_20 = torch.ops.aten.mean.dim(convolution_default_38, [0, 2, 3], True)
        add_tensor_88 = torch.ops.aten.add.Tensor(var_correction_40, 0.001)
        var_correction_40 = None
        sqrt_default_20 = torch.ops.aten.sqrt.default(add_tensor_88)
        add_tensor_88 = None
        reciprocal_default_20 = torch.ops.aten.reciprocal.default(sqrt_default_20)
        sqrt_default_20 = None
        sub_tensor_20 = torch.ops.aten.sub.Tensor(convolution_default_38, mean_dim_20)
        mul_tensor_140 = torch.ops.aten.mul.Tensor(sub_tensor_20, reciprocal_default_20)
        sub_tensor_20 = None
        squeeze_dim_120 = torch.ops.aten.squeeze.dim(mean_dim_20, 3)
        mean_dim_20 = None
        squeeze_dim_121 = torch.ops.aten.squeeze.dim(squeeze_dim_120, 2)
        squeeze_dim_120 = None
        squeeze_dim_122 = torch.ops.aten.squeeze.dim(squeeze_dim_121, 0)
        squeeze_dim_121 = None
        squeeze_dim_123 = torch.ops.aten.squeeze.dim(reciprocal_default_20, 3)
        reciprocal_default_20 = None
        squeeze_dim_124 = torch.ops.aten.squeeze.dim(squeeze_dim_123, 2)
        squeeze_dim_123 = None
        squeeze_dim_125 = torch.ops.aten.squeeze.dim(squeeze_dim_124, 0)
        squeeze_dim_124 = None
        unsqueeze_default_80 = torch.ops.aten.unsqueeze.default(primals_86, -1)
        unsqueeze_default_81 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_80, -1
        )
        unsqueeze_default_80 = None
        unsqueeze_default_82 = torch.ops.aten.unsqueeze.default(primals_87, -1)
        primals_87 = None
        unsqueeze_default_83 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_82, -1
        )
        unsqueeze_default_82 = None
        mul_tensor_146 = torch.ops.aten.mul.Tensor(mul_tensor_140, unsqueeze_default_81)
        mul_tensor_140 = unsqueeze_default_81 = None
        add_tensor_91 = torch.ops.aten.add.Tensor(mul_tensor_146, unsqueeze_default_83)
        mul_tensor_146 = unsqueeze_default_83 = None
        relu_default_20 = torch.ops.aten.relu.default(add_tensor_91)
        add_tensor_91 = None
        convolution_default_39 = torch.ops.aten.convolution.default(
            relu_default_20, primals_88, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 84
        )
        convolution_default_40 = torch.ops.aten.convolution.default(
            convolution_default_39,
            primals_89,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_42 = torch.ops.aten.var.correction(
            convolution_default_40, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_21 = torch.ops.aten.mean.dim(convolution_default_40, [0, 2, 3], True)
        add_tensor_92 = torch.ops.aten.add.Tensor(var_correction_42, 0.001)
        var_correction_42 = None
        sqrt_default_21 = torch.ops.aten.sqrt.default(add_tensor_92)
        add_tensor_92 = None
        reciprocal_default_21 = torch.ops.aten.reciprocal.default(sqrt_default_21)
        sqrt_default_21 = None
        sub_tensor_21 = torch.ops.aten.sub.Tensor(convolution_default_40, mean_dim_21)
        mul_tensor_147 = torch.ops.aten.mul.Tensor(sub_tensor_21, reciprocal_default_21)
        sub_tensor_21 = None
        squeeze_dim_126 = torch.ops.aten.squeeze.dim(mean_dim_21, 3)
        mean_dim_21 = None
        squeeze_dim_127 = torch.ops.aten.squeeze.dim(squeeze_dim_126, 2)
        squeeze_dim_126 = None
        squeeze_dim_128 = torch.ops.aten.squeeze.dim(squeeze_dim_127, 0)
        squeeze_dim_127 = None
        squeeze_dim_129 = torch.ops.aten.squeeze.dim(reciprocal_default_21, 3)
        reciprocal_default_21 = None
        squeeze_dim_130 = torch.ops.aten.squeeze.dim(squeeze_dim_129, 2)
        squeeze_dim_129 = None
        squeeze_dim_131 = torch.ops.aten.squeeze.dim(squeeze_dim_130, 0)
        squeeze_dim_130 = None
        unsqueeze_default_84 = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_default_85 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_84, -1
        )
        unsqueeze_default_84 = None
        unsqueeze_default_86 = torch.ops.aten.unsqueeze.default(primals_91, -1)
        primals_91 = None
        unsqueeze_default_87 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_86, -1
        )
        unsqueeze_default_86 = None
        mul_tensor_153 = torch.ops.aten.mul.Tensor(mul_tensor_147, unsqueeze_default_85)
        mul_tensor_147 = unsqueeze_default_85 = None
        add_tensor_95 = torch.ops.aten.add.Tensor(mul_tensor_153, unsqueeze_default_87)
        mul_tensor_153 = unsqueeze_default_87 = None
        add_tensor_96 = torch.ops.aten.add.Tensor(avg_pool2d_default_4, add_tensor_95)
        avg_pool2d_default_4 = add_tensor_95 = None
        avg_pool2d_default_5 = torch.ops.aten.avg_pool2d.default(
            add_tensor_78, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_97 = torch.ops.aten.add.Tensor(avg_pool2d_default_5, add_tensor_87)
        avg_pool2d_default_5 = None
        relu_default_21 = torch.ops.aten.relu.default(add_tensor_78)
        convolution_default_41 = torch.ops.aten.convolution.default(
            relu_default_21, primals_92, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 84
        )
        relu_default_21 = None
        convolution_default_42 = torch.ops.aten.convolution.default(
            convolution_default_41,
            primals_93,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_44 = torch.ops.aten.var.correction(
            convolution_default_42, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_22 = torch.ops.aten.mean.dim(convolution_default_42, [0, 2, 3], True)
        add_tensor_98 = torch.ops.aten.add.Tensor(var_correction_44, 0.001)
        var_correction_44 = None
        sqrt_default_22 = torch.ops.aten.sqrt.default(add_tensor_98)
        add_tensor_98 = None
        reciprocal_default_22 = torch.ops.aten.reciprocal.default(sqrt_default_22)
        sqrt_default_22 = None
        sub_tensor_22 = torch.ops.aten.sub.Tensor(convolution_default_42, mean_dim_22)
        mul_tensor_154 = torch.ops.aten.mul.Tensor(sub_tensor_22, reciprocal_default_22)
        sub_tensor_22 = None
        squeeze_dim_132 = torch.ops.aten.squeeze.dim(mean_dim_22, 3)
        mean_dim_22 = None
        squeeze_dim_133 = torch.ops.aten.squeeze.dim(squeeze_dim_132, 2)
        squeeze_dim_132 = None
        squeeze_dim_134 = torch.ops.aten.squeeze.dim(squeeze_dim_133, 0)
        squeeze_dim_133 = None
        squeeze_dim_135 = torch.ops.aten.squeeze.dim(reciprocal_default_22, 3)
        reciprocal_default_22 = None
        squeeze_dim_136 = torch.ops.aten.squeeze.dim(squeeze_dim_135, 2)
        squeeze_dim_135 = None
        squeeze_dim_137 = torch.ops.aten.squeeze.dim(squeeze_dim_136, 0)
        squeeze_dim_136 = None
        unsqueeze_default_88 = torch.ops.aten.unsqueeze.default(primals_94, -1)
        unsqueeze_default_89 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_88, -1
        )
        unsqueeze_default_88 = None
        unsqueeze_default_90 = torch.ops.aten.unsqueeze.default(primals_95, -1)
        primals_95 = None
        unsqueeze_default_91 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_90, -1
        )
        unsqueeze_default_90 = None
        mul_tensor_160 = torch.ops.aten.mul.Tensor(mul_tensor_154, unsqueeze_default_89)
        mul_tensor_154 = unsqueeze_default_89 = None
        add_tensor_101 = torch.ops.aten.add.Tensor(mul_tensor_160, unsqueeze_default_91)
        mul_tensor_160 = unsqueeze_default_91 = None
        relu_default_22 = torch.ops.aten.relu.default(add_tensor_101)
        add_tensor_101 = None
        convolution_default_43 = torch.ops.aten.convolution.default(
            relu_default_22, primals_96, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 84
        )
        convolution_default_44 = torch.ops.aten.convolution.default(
            convolution_default_43,
            primals_97,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_46 = torch.ops.aten.var.correction(
            convolution_default_44, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_23 = torch.ops.aten.mean.dim(convolution_default_44, [0, 2, 3], True)
        add_tensor_102 = torch.ops.aten.add.Tensor(var_correction_46, 0.001)
        var_correction_46 = None
        sqrt_default_23 = torch.ops.aten.sqrt.default(add_tensor_102)
        add_tensor_102 = None
        reciprocal_default_23 = torch.ops.aten.reciprocal.default(sqrt_default_23)
        sqrt_default_23 = None
        sub_tensor_23 = torch.ops.aten.sub.Tensor(convolution_default_44, mean_dim_23)
        mul_tensor_161 = torch.ops.aten.mul.Tensor(sub_tensor_23, reciprocal_default_23)
        sub_tensor_23 = None
        squeeze_dim_138 = torch.ops.aten.squeeze.dim(mean_dim_23, 3)
        mean_dim_23 = None
        squeeze_dim_139 = torch.ops.aten.squeeze.dim(squeeze_dim_138, 2)
        squeeze_dim_138 = None
        squeeze_dim_140 = torch.ops.aten.squeeze.dim(squeeze_dim_139, 0)
        squeeze_dim_139 = None
        squeeze_dim_141 = torch.ops.aten.squeeze.dim(reciprocal_default_23, 3)
        reciprocal_default_23 = None
        squeeze_dim_142 = torch.ops.aten.squeeze.dim(squeeze_dim_141, 2)
        squeeze_dim_141 = None
        squeeze_dim_143 = torch.ops.aten.squeeze.dim(squeeze_dim_142, 0)
        squeeze_dim_142 = None
        unsqueeze_default_92 = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_default_93 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_92, -1
        )
        unsqueeze_default_92 = None
        unsqueeze_default_94 = torch.ops.aten.unsqueeze.default(primals_99, -1)
        primals_99 = None
        unsqueeze_default_95 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_94, -1
        )
        unsqueeze_default_94 = None
        mul_tensor_167 = torch.ops.aten.mul.Tensor(mul_tensor_161, unsqueeze_default_93)
        mul_tensor_161 = unsqueeze_default_93 = None
        add_tensor_105 = torch.ops.aten.add.Tensor(mul_tensor_167, unsqueeze_default_95)
        mul_tensor_167 = unsqueeze_default_95 = None
        add_tensor_106 = torch.ops.aten.add.Tensor(add_tensor_105, getitem_4)
        add_tensor_105 = getitem_4 = None
        cat_default_2 = torch.ops.aten.cat.default(
            [add_tensor_87, add_tensor_96, add_tensor_97, add_tensor_106], 1
        )
        add_tensor_87 = add_tensor_96 = add_tensor_97 = add_tensor_106 = None
        avg_pool2d_default_6 = torch.ops.aten.avg_pool2d.default(
            relu_default_11, [1, 1], [2, 2], [0, 0], False, False
        )
        convolution_default_45 = torch.ops.aten.convolution.default(
            avg_pool2d_default_6,
            primals_100,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        constant_pad_nd_default_15 = torch.ops.aten.constant_pad_nd.default(
            relu_default_11, [-1, 1, -1, 1], 0.0
        )
        avg_pool2d_default_7 = torch.ops.aten.avg_pool2d.default(
            constant_pad_nd_default_15, [1, 1], [2, 2], [0, 0], False, False
        )
        convolution_default_46 = torch.ops.aten.convolution.default(
            avg_pool2d_default_7,
            primals_101,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        cat_default_3 = torch.ops.aten.cat.default(
            [convolution_default_45, convolution_default_46], 1
        )
        convolution_default_45 = convolution_default_46 = None
        var_correction_48 = torch.ops.aten.var.correction(
            cat_default_3, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_24 = torch.ops.aten.mean.dim(cat_default_3, [0, 2, 3], True)
        add_tensor_107 = torch.ops.aten.add.Tensor(var_correction_48, 0.001)
        var_correction_48 = None
        sqrt_default_24 = torch.ops.aten.sqrt.default(add_tensor_107)
        add_tensor_107 = None
        reciprocal_default_24 = torch.ops.aten.reciprocal.default(sqrt_default_24)
        sqrt_default_24 = None
        sub_tensor_24 = torch.ops.aten.sub.Tensor(cat_default_3, mean_dim_24)
        mul_tensor_168 = torch.ops.aten.mul.Tensor(sub_tensor_24, reciprocal_default_24)
        sub_tensor_24 = None
        unsqueeze_default_96 = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_default_97 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_96, -1
        )
        unsqueeze_default_96 = None
        unsqueeze_default_98 = torch.ops.aten.unsqueeze.default(primals_103, -1)
        unsqueeze_default_99 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_98, -1
        )
        unsqueeze_default_98 = None
        mul_tensor_174 = torch.ops.aten.mul.Tensor(mul_tensor_168, unsqueeze_default_97)
        mul_tensor_168 = unsqueeze_default_97 = None
        add_tensor_110 = torch.ops.aten.add.Tensor(mul_tensor_174, unsqueeze_default_99)
        mul_tensor_174 = unsqueeze_default_99 = None
        relu_default_24 = torch.ops.aten.relu.default(cat_default_2)
        cat_default_2 = None
        convolution_default_47 = torch.ops.aten.convolution.default(
            relu_default_24, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_50 = torch.ops.aten.var.correction(
            convolution_default_47, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_25 = torch.ops.aten.mean.dim(convolution_default_47, [0, 2, 3], True)
        add_tensor_111 = torch.ops.aten.add.Tensor(var_correction_50, 0.001)
        var_correction_50 = None
        sqrt_default_25 = torch.ops.aten.sqrt.default(add_tensor_111)
        add_tensor_111 = None
        reciprocal_default_25 = torch.ops.aten.reciprocal.default(sqrt_default_25)
        sqrt_default_25 = None
        sub_tensor_25 = torch.ops.aten.sub.Tensor(convolution_default_47, mean_dim_25)
        mul_tensor_175 = torch.ops.aten.mul.Tensor(sub_tensor_25, reciprocal_default_25)
        sub_tensor_25 = None
        unsqueeze_default_100 = torch.ops.aten.unsqueeze.default(primals_105, -1)
        unsqueeze_default_101 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_100, -1
        )
        unsqueeze_default_100 = None
        unsqueeze_default_102 = torch.ops.aten.unsqueeze.default(primals_106, -1)
        unsqueeze_default_103 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_102, -1
        )
        unsqueeze_default_102 = None
        mul_tensor_181 = torch.ops.aten.mul.Tensor(
            mul_tensor_175, unsqueeze_default_101
        )
        mul_tensor_175 = unsqueeze_default_101 = None
        add_tensor_114 = torch.ops.aten.add.Tensor(
            mul_tensor_181, unsqueeze_default_103
        )
        mul_tensor_181 = unsqueeze_default_103 = None
        relu_default_25 = torch.ops.aten.relu.default(add_tensor_114)
        convolution_default_48 = torch.ops.aten.convolution.default(
            relu_default_25,
            primals_107,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_49 = torch.ops.aten.convolution.default(
            convolution_default_48,
            primals_108,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_52 = torch.ops.aten.var.correction(
            convolution_default_49, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_26 = torch.ops.aten.mean.dim(convolution_default_49, [0, 2, 3], True)
        add_tensor_115 = torch.ops.aten.add.Tensor(var_correction_52, 0.001)
        var_correction_52 = None
        sqrt_default_26 = torch.ops.aten.sqrt.default(add_tensor_115)
        add_tensor_115 = None
        reciprocal_default_26 = torch.ops.aten.reciprocal.default(sqrt_default_26)
        sqrt_default_26 = None
        sub_tensor_26 = torch.ops.aten.sub.Tensor(convolution_default_49, mean_dim_26)
        mul_tensor_182 = torch.ops.aten.mul.Tensor(sub_tensor_26, reciprocal_default_26)
        sub_tensor_26 = None
        squeeze_dim_156 = torch.ops.aten.squeeze.dim(mean_dim_26, 3)
        mean_dim_26 = None
        squeeze_dim_157 = torch.ops.aten.squeeze.dim(squeeze_dim_156, 2)
        squeeze_dim_156 = None
        squeeze_dim_158 = torch.ops.aten.squeeze.dim(squeeze_dim_157, 0)
        squeeze_dim_157 = None
        squeeze_dim_159 = torch.ops.aten.squeeze.dim(reciprocal_default_26, 3)
        reciprocal_default_26 = None
        squeeze_dim_160 = torch.ops.aten.squeeze.dim(squeeze_dim_159, 2)
        squeeze_dim_159 = None
        squeeze_dim_161 = torch.ops.aten.squeeze.dim(squeeze_dim_160, 0)
        squeeze_dim_160 = None
        unsqueeze_default_104 = torch.ops.aten.unsqueeze.default(primals_109, -1)
        unsqueeze_default_105 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_104, -1
        )
        unsqueeze_default_104 = None
        unsqueeze_default_106 = torch.ops.aten.unsqueeze.default(primals_110, -1)
        primals_110 = None
        unsqueeze_default_107 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_106, -1
        )
        unsqueeze_default_106 = None
        mul_tensor_188 = torch.ops.aten.mul.Tensor(
            mul_tensor_182, unsqueeze_default_105
        )
        mul_tensor_182 = unsqueeze_default_105 = None
        add_tensor_118 = torch.ops.aten.add.Tensor(
            mul_tensor_188, unsqueeze_default_107
        )
        mul_tensor_188 = unsqueeze_default_107 = None
        relu_default_26 = torch.ops.aten.relu.default(add_tensor_118)
        add_tensor_118 = None
        convolution_default_50 = torch.ops.aten.convolution.default(
            relu_default_26,
            primals_111,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_51 = torch.ops.aten.convolution.default(
            convolution_default_50,
            primals_112,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_54 = torch.ops.aten.var.correction(
            convolution_default_51, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_27 = torch.ops.aten.mean.dim(convolution_default_51, [0, 2, 3], True)
        add_tensor_119 = torch.ops.aten.add.Tensor(var_correction_54, 0.001)
        var_correction_54 = None
        sqrt_default_27 = torch.ops.aten.sqrt.default(add_tensor_119)
        add_tensor_119 = None
        reciprocal_default_27 = torch.ops.aten.reciprocal.default(sqrt_default_27)
        sqrt_default_27 = None
        sub_tensor_27 = torch.ops.aten.sub.Tensor(convolution_default_51, mean_dim_27)
        mul_tensor_189 = torch.ops.aten.mul.Tensor(sub_tensor_27, reciprocal_default_27)
        sub_tensor_27 = None
        squeeze_dim_162 = torch.ops.aten.squeeze.dim(mean_dim_27, 3)
        mean_dim_27 = None
        squeeze_dim_163 = torch.ops.aten.squeeze.dim(squeeze_dim_162, 2)
        squeeze_dim_162 = None
        squeeze_dim_164 = torch.ops.aten.squeeze.dim(squeeze_dim_163, 0)
        squeeze_dim_163 = None
        squeeze_dim_165 = torch.ops.aten.squeeze.dim(reciprocal_default_27, 3)
        reciprocal_default_27 = None
        squeeze_dim_166 = torch.ops.aten.squeeze.dim(squeeze_dim_165, 2)
        squeeze_dim_165 = None
        squeeze_dim_167 = torch.ops.aten.squeeze.dim(squeeze_dim_166, 0)
        squeeze_dim_166 = None
        unsqueeze_default_108 = torch.ops.aten.unsqueeze.default(primals_113, -1)
        unsqueeze_default_109 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_108, -1
        )
        unsqueeze_default_108 = None
        unsqueeze_default_110 = torch.ops.aten.unsqueeze.default(primals_114, -1)
        primals_114 = None
        unsqueeze_default_111 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_110, -1
        )
        unsqueeze_default_110 = None
        mul_tensor_195 = torch.ops.aten.mul.Tensor(
            mul_tensor_189, unsqueeze_default_109
        )
        mul_tensor_189 = unsqueeze_default_109 = None
        add_tensor_122 = torch.ops.aten.add.Tensor(
            mul_tensor_195, unsqueeze_default_111
        )
        mul_tensor_195 = unsqueeze_default_111 = None
        relu_default_27 = torch.ops.aten.relu.default(add_tensor_110)
        convolution_default_52 = torch.ops.aten.convolution.default(
            relu_default_27,
            primals_115,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_53 = torch.ops.aten.convolution.default(
            convolution_default_52,
            primals_116,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_56 = torch.ops.aten.var.correction(
            convolution_default_53, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_28 = torch.ops.aten.mean.dim(convolution_default_53, [0, 2, 3], True)
        add_tensor_123 = torch.ops.aten.add.Tensor(var_correction_56, 0.001)
        var_correction_56 = None
        sqrt_default_28 = torch.ops.aten.sqrt.default(add_tensor_123)
        add_tensor_123 = None
        reciprocal_default_28 = torch.ops.aten.reciprocal.default(sqrt_default_28)
        sqrt_default_28 = None
        sub_tensor_28 = torch.ops.aten.sub.Tensor(convolution_default_53, mean_dim_28)
        mul_tensor_196 = torch.ops.aten.mul.Tensor(sub_tensor_28, reciprocal_default_28)
        sub_tensor_28 = None
        squeeze_dim_168 = torch.ops.aten.squeeze.dim(mean_dim_28, 3)
        mean_dim_28 = None
        squeeze_dim_169 = torch.ops.aten.squeeze.dim(squeeze_dim_168, 2)
        squeeze_dim_168 = None
        squeeze_dim_170 = torch.ops.aten.squeeze.dim(squeeze_dim_169, 0)
        squeeze_dim_169 = None
        squeeze_dim_171 = torch.ops.aten.squeeze.dim(reciprocal_default_28, 3)
        reciprocal_default_28 = None
        squeeze_dim_172 = torch.ops.aten.squeeze.dim(squeeze_dim_171, 2)
        squeeze_dim_171 = None
        squeeze_dim_173 = torch.ops.aten.squeeze.dim(squeeze_dim_172, 0)
        squeeze_dim_172 = None
        unsqueeze_default_112 = torch.ops.aten.unsqueeze.default(primals_117, -1)
        unsqueeze_default_113 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_112, -1
        )
        unsqueeze_default_112 = None
        unsqueeze_default_114 = torch.ops.aten.unsqueeze.default(primals_118, -1)
        primals_118 = None
        unsqueeze_default_115 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_114, -1
        )
        unsqueeze_default_114 = None
        mul_tensor_202 = torch.ops.aten.mul.Tensor(
            mul_tensor_196, unsqueeze_default_113
        )
        mul_tensor_196 = unsqueeze_default_113 = None
        add_tensor_126 = torch.ops.aten.add.Tensor(
            mul_tensor_202, unsqueeze_default_115
        )
        mul_tensor_202 = unsqueeze_default_115 = None
        relu_default_28 = torch.ops.aten.relu.default(add_tensor_126)
        add_tensor_126 = None
        convolution_default_54 = torch.ops.aten.convolution.default(
            relu_default_28,
            primals_119,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_55 = torch.ops.aten.convolution.default(
            convolution_default_54,
            primals_120,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_58 = torch.ops.aten.var.correction(
            convolution_default_55, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_29 = torch.ops.aten.mean.dim(convolution_default_55, [0, 2, 3], True)
        add_tensor_127 = torch.ops.aten.add.Tensor(var_correction_58, 0.001)
        var_correction_58 = None
        sqrt_default_29 = torch.ops.aten.sqrt.default(add_tensor_127)
        add_tensor_127 = None
        reciprocal_default_29 = torch.ops.aten.reciprocal.default(sqrt_default_29)
        sqrt_default_29 = None
        sub_tensor_29 = torch.ops.aten.sub.Tensor(convolution_default_55, mean_dim_29)
        mul_tensor_203 = torch.ops.aten.mul.Tensor(sub_tensor_29, reciprocal_default_29)
        sub_tensor_29 = None
        squeeze_dim_174 = torch.ops.aten.squeeze.dim(mean_dim_29, 3)
        mean_dim_29 = None
        squeeze_dim_175 = torch.ops.aten.squeeze.dim(squeeze_dim_174, 2)
        squeeze_dim_174 = None
        squeeze_dim_176 = torch.ops.aten.squeeze.dim(squeeze_dim_175, 0)
        squeeze_dim_175 = None
        squeeze_dim_177 = torch.ops.aten.squeeze.dim(reciprocal_default_29, 3)
        reciprocal_default_29 = None
        squeeze_dim_178 = torch.ops.aten.squeeze.dim(squeeze_dim_177, 2)
        squeeze_dim_177 = None
        squeeze_dim_179 = torch.ops.aten.squeeze.dim(squeeze_dim_178, 0)
        squeeze_dim_178 = None
        unsqueeze_default_116 = torch.ops.aten.unsqueeze.default(primals_121, -1)
        unsqueeze_default_117 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_116, -1
        )
        unsqueeze_default_116 = None
        unsqueeze_default_118 = torch.ops.aten.unsqueeze.default(primals_122, -1)
        primals_122 = None
        unsqueeze_default_119 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_118, -1
        )
        unsqueeze_default_118 = None
        mul_tensor_209 = torch.ops.aten.mul.Tensor(
            mul_tensor_203, unsqueeze_default_117
        )
        mul_tensor_203 = unsqueeze_default_117 = None
        add_tensor_130 = torch.ops.aten.add.Tensor(
            mul_tensor_209, unsqueeze_default_119
        )
        mul_tensor_209 = unsqueeze_default_119 = None
        add_tensor_131 = torch.ops.aten.add.Tensor(add_tensor_122, add_tensor_130)
        add_tensor_122 = add_tensor_130 = None
        convolution_default_56 = torch.ops.aten.convolution.default(
            relu_default_27,
            primals_123,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_57 = torch.ops.aten.convolution.default(
            convolution_default_56,
            primals_124,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_60 = torch.ops.aten.var.correction(
            convolution_default_57, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_30 = torch.ops.aten.mean.dim(convolution_default_57, [0, 2, 3], True)
        add_tensor_132 = torch.ops.aten.add.Tensor(var_correction_60, 0.001)
        var_correction_60 = None
        sqrt_default_30 = torch.ops.aten.sqrt.default(add_tensor_132)
        add_tensor_132 = None
        reciprocal_default_30 = torch.ops.aten.reciprocal.default(sqrt_default_30)
        sqrt_default_30 = None
        sub_tensor_30 = torch.ops.aten.sub.Tensor(convolution_default_57, mean_dim_30)
        mul_tensor_210 = torch.ops.aten.mul.Tensor(sub_tensor_30, reciprocal_default_30)
        sub_tensor_30 = None
        squeeze_dim_180 = torch.ops.aten.squeeze.dim(mean_dim_30, 3)
        mean_dim_30 = None
        squeeze_dim_181 = torch.ops.aten.squeeze.dim(squeeze_dim_180, 2)
        squeeze_dim_180 = None
        squeeze_dim_182 = torch.ops.aten.squeeze.dim(squeeze_dim_181, 0)
        squeeze_dim_181 = None
        squeeze_dim_183 = torch.ops.aten.squeeze.dim(reciprocal_default_30, 3)
        reciprocal_default_30 = None
        squeeze_dim_184 = torch.ops.aten.squeeze.dim(squeeze_dim_183, 2)
        squeeze_dim_183 = None
        squeeze_dim_185 = torch.ops.aten.squeeze.dim(squeeze_dim_184, 0)
        squeeze_dim_184 = None
        unsqueeze_default_120 = torch.ops.aten.unsqueeze.default(primals_125, -1)
        unsqueeze_default_121 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_120, -1
        )
        unsqueeze_default_120 = None
        unsqueeze_default_122 = torch.ops.aten.unsqueeze.default(primals_126, -1)
        primals_126 = None
        unsqueeze_default_123 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_122, -1
        )
        unsqueeze_default_122 = None
        mul_tensor_216 = torch.ops.aten.mul.Tensor(
            mul_tensor_210, unsqueeze_default_121
        )
        mul_tensor_210 = unsqueeze_default_121 = None
        add_tensor_135 = torch.ops.aten.add.Tensor(
            mul_tensor_216, unsqueeze_default_123
        )
        mul_tensor_216 = unsqueeze_default_123 = None
        relu_default_30 = torch.ops.aten.relu.default(add_tensor_135)
        add_tensor_135 = None
        convolution_default_58 = torch.ops.aten.convolution.default(
            relu_default_30,
            primals_127,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_59 = torch.ops.aten.convolution.default(
            convolution_default_58,
            primals_128,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_62 = torch.ops.aten.var.correction(
            convolution_default_59, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_31 = torch.ops.aten.mean.dim(convolution_default_59, [0, 2, 3], True)
        add_tensor_136 = torch.ops.aten.add.Tensor(var_correction_62, 0.001)
        var_correction_62 = None
        sqrt_default_31 = torch.ops.aten.sqrt.default(add_tensor_136)
        add_tensor_136 = None
        reciprocal_default_31 = torch.ops.aten.reciprocal.default(sqrt_default_31)
        sqrt_default_31 = None
        sub_tensor_31 = torch.ops.aten.sub.Tensor(convolution_default_59, mean_dim_31)
        mul_tensor_217 = torch.ops.aten.mul.Tensor(sub_tensor_31, reciprocal_default_31)
        sub_tensor_31 = None
        squeeze_dim_186 = torch.ops.aten.squeeze.dim(mean_dim_31, 3)
        mean_dim_31 = None
        squeeze_dim_187 = torch.ops.aten.squeeze.dim(squeeze_dim_186, 2)
        squeeze_dim_186 = None
        squeeze_dim_188 = torch.ops.aten.squeeze.dim(squeeze_dim_187, 0)
        squeeze_dim_187 = None
        squeeze_dim_189 = torch.ops.aten.squeeze.dim(reciprocal_default_31, 3)
        reciprocal_default_31 = None
        squeeze_dim_190 = torch.ops.aten.squeeze.dim(squeeze_dim_189, 2)
        squeeze_dim_189 = None
        squeeze_dim_191 = torch.ops.aten.squeeze.dim(squeeze_dim_190, 0)
        squeeze_dim_190 = None
        unsqueeze_default_124 = torch.ops.aten.unsqueeze.default(primals_129, -1)
        unsqueeze_default_125 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_124, -1
        )
        unsqueeze_default_124 = None
        unsqueeze_default_126 = torch.ops.aten.unsqueeze.default(primals_130, -1)
        primals_130 = None
        unsqueeze_default_127 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_126, -1
        )
        unsqueeze_default_126 = None
        mul_tensor_223 = torch.ops.aten.mul.Tensor(
            mul_tensor_217, unsqueeze_default_125
        )
        mul_tensor_217 = unsqueeze_default_125 = None
        add_tensor_139 = torch.ops.aten.add.Tensor(
            mul_tensor_223, unsqueeze_default_127
        )
        mul_tensor_223 = unsqueeze_default_127 = None
        convolution_default_60 = torch.ops.aten.convolution.default(
            relu_default_27,
            primals_131,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_27 = None
        convolution_default_61 = torch.ops.aten.convolution.default(
            convolution_default_60,
            primals_132,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_64 = torch.ops.aten.var.correction(
            convolution_default_61, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_32 = torch.ops.aten.mean.dim(convolution_default_61, [0, 2, 3], True)
        add_tensor_140 = torch.ops.aten.add.Tensor(var_correction_64, 0.001)
        var_correction_64 = None
        sqrt_default_32 = torch.ops.aten.sqrt.default(add_tensor_140)
        add_tensor_140 = None
        reciprocal_default_32 = torch.ops.aten.reciprocal.default(sqrt_default_32)
        sqrt_default_32 = None
        sub_tensor_32 = torch.ops.aten.sub.Tensor(convolution_default_61, mean_dim_32)
        mul_tensor_224 = torch.ops.aten.mul.Tensor(sub_tensor_32, reciprocal_default_32)
        sub_tensor_32 = None
        squeeze_dim_192 = torch.ops.aten.squeeze.dim(mean_dim_32, 3)
        mean_dim_32 = None
        squeeze_dim_193 = torch.ops.aten.squeeze.dim(squeeze_dim_192, 2)
        squeeze_dim_192 = None
        squeeze_dim_194 = torch.ops.aten.squeeze.dim(squeeze_dim_193, 0)
        squeeze_dim_193 = None
        squeeze_dim_195 = torch.ops.aten.squeeze.dim(reciprocal_default_32, 3)
        reciprocal_default_32 = None
        squeeze_dim_196 = torch.ops.aten.squeeze.dim(squeeze_dim_195, 2)
        squeeze_dim_195 = None
        squeeze_dim_197 = torch.ops.aten.squeeze.dim(squeeze_dim_196, 0)
        squeeze_dim_196 = None
        unsqueeze_default_128 = torch.ops.aten.unsqueeze.default(primals_133, -1)
        unsqueeze_default_129 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_128, -1
        )
        unsqueeze_default_128 = None
        unsqueeze_default_130 = torch.ops.aten.unsqueeze.default(primals_134, -1)
        primals_134 = None
        unsqueeze_default_131 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_130, -1
        )
        unsqueeze_default_130 = None
        mul_tensor_230 = torch.ops.aten.mul.Tensor(
            mul_tensor_224, unsqueeze_default_129
        )
        mul_tensor_224 = unsqueeze_default_129 = None
        add_tensor_143 = torch.ops.aten.add.Tensor(
            mul_tensor_230, unsqueeze_default_131
        )
        mul_tensor_230 = unsqueeze_default_131 = None
        relu_default_32 = torch.ops.aten.relu.default(add_tensor_143)
        add_tensor_143 = None
        convolution_default_62 = torch.ops.aten.convolution.default(
            relu_default_32,
            primals_135,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_63 = torch.ops.aten.convolution.default(
            convolution_default_62,
            primals_136,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_66 = torch.ops.aten.var.correction(
            convolution_default_63, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_33 = torch.ops.aten.mean.dim(convolution_default_63, [0, 2, 3], True)
        add_tensor_144 = torch.ops.aten.add.Tensor(var_correction_66, 0.001)
        var_correction_66 = None
        sqrt_default_33 = torch.ops.aten.sqrt.default(add_tensor_144)
        add_tensor_144 = None
        reciprocal_default_33 = torch.ops.aten.reciprocal.default(sqrt_default_33)
        sqrt_default_33 = None
        sub_tensor_33 = torch.ops.aten.sub.Tensor(convolution_default_63, mean_dim_33)
        mul_tensor_231 = torch.ops.aten.mul.Tensor(sub_tensor_33, reciprocal_default_33)
        sub_tensor_33 = None
        squeeze_dim_198 = torch.ops.aten.squeeze.dim(mean_dim_33, 3)
        mean_dim_33 = None
        squeeze_dim_199 = torch.ops.aten.squeeze.dim(squeeze_dim_198, 2)
        squeeze_dim_198 = None
        squeeze_dim_200 = torch.ops.aten.squeeze.dim(squeeze_dim_199, 0)
        squeeze_dim_199 = None
        squeeze_dim_201 = torch.ops.aten.squeeze.dim(reciprocal_default_33, 3)
        reciprocal_default_33 = None
        squeeze_dim_202 = torch.ops.aten.squeeze.dim(squeeze_dim_201, 2)
        squeeze_dim_201 = None
        squeeze_dim_203 = torch.ops.aten.squeeze.dim(squeeze_dim_202, 0)
        squeeze_dim_202 = None
        unsqueeze_default_132 = torch.ops.aten.unsqueeze.default(primals_137, -1)
        unsqueeze_default_133 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_132, -1
        )
        unsqueeze_default_132 = None
        unsqueeze_default_134 = torch.ops.aten.unsqueeze.default(primals_138, -1)
        primals_138 = None
        unsqueeze_default_135 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_134, -1
        )
        unsqueeze_default_134 = None
        mul_tensor_237 = torch.ops.aten.mul.Tensor(
            mul_tensor_231, unsqueeze_default_133
        )
        mul_tensor_231 = unsqueeze_default_133 = None
        add_tensor_147 = torch.ops.aten.add.Tensor(
            mul_tensor_237, unsqueeze_default_135
        )
        mul_tensor_237 = unsqueeze_default_135 = None
        add_tensor_148 = torch.ops.aten.add.Tensor(add_tensor_139, add_tensor_147)
        add_tensor_139 = add_tensor_147 = None
        avg_pool2d_default_8 = torch.ops.aten.avg_pool2d.default(
            add_tensor_114, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_149 = torch.ops.aten.add.Tensor(avg_pool2d_default_8, add_tensor_110)
        avg_pool2d_default_8 = None
        avg_pool2d_default_9 = torch.ops.aten.avg_pool2d.default(
            add_tensor_110, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_150 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_9, avg_pool2d_default_9
        )
        avg_pool2d_default_9 = None
        convolution_default_64 = torch.ops.aten.convolution.default(
            relu_default_25,
            primals_139,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_25 = None
        convolution_default_65 = torch.ops.aten.convolution.default(
            convolution_default_64,
            primals_140,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_68 = torch.ops.aten.var.correction(
            convolution_default_65, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_34 = torch.ops.aten.mean.dim(convolution_default_65, [0, 2, 3], True)
        add_tensor_151 = torch.ops.aten.add.Tensor(var_correction_68, 0.001)
        var_correction_68 = None
        sqrt_default_34 = torch.ops.aten.sqrt.default(add_tensor_151)
        add_tensor_151 = None
        reciprocal_default_34 = torch.ops.aten.reciprocal.default(sqrt_default_34)
        sqrt_default_34 = None
        sub_tensor_34 = torch.ops.aten.sub.Tensor(convolution_default_65, mean_dim_34)
        mul_tensor_238 = torch.ops.aten.mul.Tensor(sub_tensor_34, reciprocal_default_34)
        sub_tensor_34 = None
        squeeze_dim_204 = torch.ops.aten.squeeze.dim(mean_dim_34, 3)
        mean_dim_34 = None
        squeeze_dim_205 = torch.ops.aten.squeeze.dim(squeeze_dim_204, 2)
        squeeze_dim_204 = None
        squeeze_dim_206 = torch.ops.aten.squeeze.dim(squeeze_dim_205, 0)
        squeeze_dim_205 = None
        squeeze_dim_207 = torch.ops.aten.squeeze.dim(reciprocal_default_34, 3)
        reciprocal_default_34 = None
        squeeze_dim_208 = torch.ops.aten.squeeze.dim(squeeze_dim_207, 2)
        squeeze_dim_207 = None
        squeeze_dim_209 = torch.ops.aten.squeeze.dim(squeeze_dim_208, 0)
        squeeze_dim_208 = None
        unsqueeze_default_136 = torch.ops.aten.unsqueeze.default(primals_141, -1)
        unsqueeze_default_137 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_136, -1
        )
        unsqueeze_default_136 = None
        unsqueeze_default_138 = torch.ops.aten.unsqueeze.default(primals_142, -1)
        primals_142 = None
        unsqueeze_default_139 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_138, -1
        )
        unsqueeze_default_138 = None
        mul_tensor_244 = torch.ops.aten.mul.Tensor(
            mul_tensor_238, unsqueeze_default_137
        )
        mul_tensor_238 = unsqueeze_default_137 = None
        add_tensor_154 = torch.ops.aten.add.Tensor(
            mul_tensor_244, unsqueeze_default_139
        )
        mul_tensor_244 = unsqueeze_default_139 = None
        relu_default_34 = torch.ops.aten.relu.default(add_tensor_154)
        add_tensor_154 = None
        convolution_default_66 = torch.ops.aten.convolution.default(
            relu_default_34,
            primals_143,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_67 = torch.ops.aten.convolution.default(
            convolution_default_66,
            primals_144,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_70 = torch.ops.aten.var.correction(
            convolution_default_67, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_35 = torch.ops.aten.mean.dim(convolution_default_67, [0, 2, 3], True)
        add_tensor_155 = torch.ops.aten.add.Tensor(var_correction_70, 0.001)
        var_correction_70 = None
        sqrt_default_35 = torch.ops.aten.sqrt.default(add_tensor_155)
        add_tensor_155 = None
        reciprocal_default_35 = torch.ops.aten.reciprocal.default(sqrt_default_35)
        sqrt_default_35 = None
        sub_tensor_35 = torch.ops.aten.sub.Tensor(convolution_default_67, mean_dim_35)
        mul_tensor_245 = torch.ops.aten.mul.Tensor(sub_tensor_35, reciprocal_default_35)
        sub_tensor_35 = None
        squeeze_dim_210 = torch.ops.aten.squeeze.dim(mean_dim_35, 3)
        mean_dim_35 = None
        squeeze_dim_211 = torch.ops.aten.squeeze.dim(squeeze_dim_210, 2)
        squeeze_dim_210 = None
        squeeze_dim_212 = torch.ops.aten.squeeze.dim(squeeze_dim_211, 0)
        squeeze_dim_211 = None
        squeeze_dim_213 = torch.ops.aten.squeeze.dim(reciprocal_default_35, 3)
        reciprocal_default_35 = None
        squeeze_dim_214 = torch.ops.aten.squeeze.dim(squeeze_dim_213, 2)
        squeeze_dim_213 = None
        squeeze_dim_215 = torch.ops.aten.squeeze.dim(squeeze_dim_214, 0)
        squeeze_dim_214 = None
        unsqueeze_default_140 = torch.ops.aten.unsqueeze.default(primals_145, -1)
        unsqueeze_default_141 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_140, -1
        )
        unsqueeze_default_140 = None
        unsqueeze_default_142 = torch.ops.aten.unsqueeze.default(primals_146, -1)
        primals_146 = None
        unsqueeze_default_143 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_142, -1
        )
        unsqueeze_default_142 = None
        mul_tensor_251 = torch.ops.aten.mul.Tensor(
            mul_tensor_245, unsqueeze_default_141
        )
        mul_tensor_245 = unsqueeze_default_141 = None
        add_tensor_158 = torch.ops.aten.add.Tensor(
            mul_tensor_251, unsqueeze_default_143
        )
        mul_tensor_251 = unsqueeze_default_143 = None
        add_tensor_159 = torch.ops.aten.add.Tensor(add_tensor_158, add_tensor_114)
        add_tensor_158 = add_tensor_114 = None
        cat_default_4 = torch.ops.aten.cat.default(
            [
                add_tensor_110,
                add_tensor_131,
                add_tensor_148,
                add_tensor_149,
                add_tensor_150,
                add_tensor_159,
            ],
            1,
        )
        add_tensor_110 = (
            add_tensor_131
        ) = add_tensor_148 = add_tensor_149 = add_tensor_150 = add_tensor_159 = None
        convolution_default_68 = torch.ops.aten.convolution.default(
            relu_default_24, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_72 = torch.ops.aten.var.correction(
            convolution_default_68, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_36 = torch.ops.aten.mean.dim(convolution_default_68, [0, 2, 3], True)
        add_tensor_160 = torch.ops.aten.add.Tensor(var_correction_72, 0.001)
        var_correction_72 = None
        sqrt_default_36 = torch.ops.aten.sqrt.default(add_tensor_160)
        add_tensor_160 = None
        reciprocal_default_36 = torch.ops.aten.reciprocal.default(sqrt_default_36)
        sqrt_default_36 = None
        sub_tensor_36 = torch.ops.aten.sub.Tensor(convolution_default_68, mean_dim_36)
        mul_tensor_252 = torch.ops.aten.mul.Tensor(sub_tensor_36, reciprocal_default_36)
        sub_tensor_36 = None
        unsqueeze_default_144 = torch.ops.aten.unsqueeze.default(primals_148, -1)
        unsqueeze_default_145 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_144, -1
        )
        unsqueeze_default_144 = None
        unsqueeze_default_146 = torch.ops.aten.unsqueeze.default(primals_149, -1)
        unsqueeze_default_147 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_146, -1
        )
        unsqueeze_default_146 = None
        mul_tensor_258 = torch.ops.aten.mul.Tensor(
            mul_tensor_252, unsqueeze_default_145
        )
        mul_tensor_252 = unsqueeze_default_145 = None
        add_tensor_163 = torch.ops.aten.add.Tensor(
            mul_tensor_258, unsqueeze_default_147
        )
        mul_tensor_258 = unsqueeze_default_147 = None
        relu_default_36 = torch.ops.aten.relu.default(cat_default_4)
        cat_default_4 = None
        convolution_default_69 = torch.ops.aten.convolution.default(
            relu_default_36, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_74 = torch.ops.aten.var.correction(
            convolution_default_69, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_37 = torch.ops.aten.mean.dim(convolution_default_69, [0, 2, 3], True)
        add_tensor_164 = torch.ops.aten.add.Tensor(var_correction_74, 0.001)
        var_correction_74 = None
        sqrt_default_37 = torch.ops.aten.sqrt.default(add_tensor_164)
        add_tensor_164 = None
        reciprocal_default_37 = torch.ops.aten.reciprocal.default(sqrt_default_37)
        sqrt_default_37 = None
        sub_tensor_37 = torch.ops.aten.sub.Tensor(convolution_default_69, mean_dim_37)
        mul_tensor_259 = torch.ops.aten.mul.Tensor(sub_tensor_37, reciprocal_default_37)
        sub_tensor_37 = None
        unsqueeze_default_148 = torch.ops.aten.unsqueeze.default(primals_151, -1)
        unsqueeze_default_149 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_148, -1
        )
        unsqueeze_default_148 = None
        unsqueeze_default_150 = torch.ops.aten.unsqueeze.default(primals_152, -1)
        unsqueeze_default_151 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_150, -1
        )
        unsqueeze_default_150 = None
        mul_tensor_265 = torch.ops.aten.mul.Tensor(
            mul_tensor_259, unsqueeze_default_149
        )
        mul_tensor_259 = unsqueeze_default_149 = None
        add_tensor_167 = torch.ops.aten.add.Tensor(
            mul_tensor_265, unsqueeze_default_151
        )
        mul_tensor_265 = unsqueeze_default_151 = None
        relu_default_37 = torch.ops.aten.relu.default(add_tensor_167)
        convolution_default_70 = torch.ops.aten.convolution.default(
            relu_default_37,
            primals_153,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_71 = torch.ops.aten.convolution.default(
            convolution_default_70,
            primals_154,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_76 = torch.ops.aten.var.correction(
            convolution_default_71, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_38 = torch.ops.aten.mean.dim(convolution_default_71, [0, 2, 3], True)
        add_tensor_168 = torch.ops.aten.add.Tensor(var_correction_76, 0.001)
        var_correction_76 = None
        sqrt_default_38 = torch.ops.aten.sqrt.default(add_tensor_168)
        add_tensor_168 = None
        reciprocal_default_38 = torch.ops.aten.reciprocal.default(sqrt_default_38)
        sqrt_default_38 = None
        sub_tensor_38 = torch.ops.aten.sub.Tensor(convolution_default_71, mean_dim_38)
        mul_tensor_266 = torch.ops.aten.mul.Tensor(sub_tensor_38, reciprocal_default_38)
        sub_tensor_38 = None
        squeeze_dim_228 = torch.ops.aten.squeeze.dim(mean_dim_38, 3)
        mean_dim_38 = None
        squeeze_dim_229 = torch.ops.aten.squeeze.dim(squeeze_dim_228, 2)
        squeeze_dim_228 = None
        squeeze_dim_230 = torch.ops.aten.squeeze.dim(squeeze_dim_229, 0)
        squeeze_dim_229 = None
        squeeze_dim_231 = torch.ops.aten.squeeze.dim(reciprocal_default_38, 3)
        reciprocal_default_38 = None
        squeeze_dim_232 = torch.ops.aten.squeeze.dim(squeeze_dim_231, 2)
        squeeze_dim_231 = None
        squeeze_dim_233 = torch.ops.aten.squeeze.dim(squeeze_dim_232, 0)
        squeeze_dim_232 = None
        unsqueeze_default_152 = torch.ops.aten.unsqueeze.default(primals_155, -1)
        unsqueeze_default_153 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_152, -1
        )
        unsqueeze_default_152 = None
        unsqueeze_default_154 = torch.ops.aten.unsqueeze.default(primals_156, -1)
        primals_156 = None
        unsqueeze_default_155 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_154, -1
        )
        unsqueeze_default_154 = None
        mul_tensor_272 = torch.ops.aten.mul.Tensor(
            mul_tensor_266, unsqueeze_default_153
        )
        mul_tensor_266 = unsqueeze_default_153 = None
        add_tensor_171 = torch.ops.aten.add.Tensor(
            mul_tensor_272, unsqueeze_default_155
        )
        mul_tensor_272 = unsqueeze_default_155 = None
        relu_default_38 = torch.ops.aten.relu.default(add_tensor_171)
        add_tensor_171 = None
        convolution_default_72 = torch.ops.aten.convolution.default(
            relu_default_38,
            primals_157,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_73 = torch.ops.aten.convolution.default(
            convolution_default_72,
            primals_158,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_78 = torch.ops.aten.var.correction(
            convolution_default_73, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_39 = torch.ops.aten.mean.dim(convolution_default_73, [0, 2, 3], True)
        add_tensor_172 = torch.ops.aten.add.Tensor(var_correction_78, 0.001)
        var_correction_78 = None
        sqrt_default_39 = torch.ops.aten.sqrt.default(add_tensor_172)
        add_tensor_172 = None
        reciprocal_default_39 = torch.ops.aten.reciprocal.default(sqrt_default_39)
        sqrt_default_39 = None
        sub_tensor_39 = torch.ops.aten.sub.Tensor(convolution_default_73, mean_dim_39)
        mul_tensor_273 = torch.ops.aten.mul.Tensor(sub_tensor_39, reciprocal_default_39)
        sub_tensor_39 = None
        squeeze_dim_234 = torch.ops.aten.squeeze.dim(mean_dim_39, 3)
        mean_dim_39 = None
        squeeze_dim_235 = torch.ops.aten.squeeze.dim(squeeze_dim_234, 2)
        squeeze_dim_234 = None
        squeeze_dim_236 = torch.ops.aten.squeeze.dim(squeeze_dim_235, 0)
        squeeze_dim_235 = None
        squeeze_dim_237 = torch.ops.aten.squeeze.dim(reciprocal_default_39, 3)
        reciprocal_default_39 = None
        squeeze_dim_238 = torch.ops.aten.squeeze.dim(squeeze_dim_237, 2)
        squeeze_dim_237 = None
        squeeze_dim_239 = torch.ops.aten.squeeze.dim(squeeze_dim_238, 0)
        squeeze_dim_238 = None
        unsqueeze_default_156 = torch.ops.aten.unsqueeze.default(primals_159, -1)
        unsqueeze_default_157 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_156, -1
        )
        unsqueeze_default_156 = None
        unsqueeze_default_158 = torch.ops.aten.unsqueeze.default(primals_160, -1)
        primals_160 = None
        unsqueeze_default_159 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_158, -1
        )
        unsqueeze_default_158 = None
        mul_tensor_279 = torch.ops.aten.mul.Tensor(
            mul_tensor_273, unsqueeze_default_157
        )
        mul_tensor_273 = unsqueeze_default_157 = None
        add_tensor_175 = torch.ops.aten.add.Tensor(
            mul_tensor_279, unsqueeze_default_159
        )
        mul_tensor_279 = unsqueeze_default_159 = None
        relu_default_39 = torch.ops.aten.relu.default(add_tensor_163)
        convolution_default_74 = torch.ops.aten.convolution.default(
            relu_default_39,
            primals_161,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_75 = torch.ops.aten.convolution.default(
            convolution_default_74,
            primals_162,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_80 = torch.ops.aten.var.correction(
            convolution_default_75, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_40 = torch.ops.aten.mean.dim(convolution_default_75, [0, 2, 3], True)
        add_tensor_176 = torch.ops.aten.add.Tensor(var_correction_80, 0.001)
        var_correction_80 = None
        sqrt_default_40 = torch.ops.aten.sqrt.default(add_tensor_176)
        add_tensor_176 = None
        reciprocal_default_40 = torch.ops.aten.reciprocal.default(sqrt_default_40)
        sqrt_default_40 = None
        sub_tensor_40 = torch.ops.aten.sub.Tensor(convolution_default_75, mean_dim_40)
        mul_tensor_280 = torch.ops.aten.mul.Tensor(sub_tensor_40, reciprocal_default_40)
        sub_tensor_40 = None
        squeeze_dim_240 = torch.ops.aten.squeeze.dim(mean_dim_40, 3)
        mean_dim_40 = None
        squeeze_dim_241 = torch.ops.aten.squeeze.dim(squeeze_dim_240, 2)
        squeeze_dim_240 = None
        squeeze_dim_242 = torch.ops.aten.squeeze.dim(squeeze_dim_241, 0)
        squeeze_dim_241 = None
        squeeze_dim_243 = torch.ops.aten.squeeze.dim(reciprocal_default_40, 3)
        reciprocal_default_40 = None
        squeeze_dim_244 = torch.ops.aten.squeeze.dim(squeeze_dim_243, 2)
        squeeze_dim_243 = None
        squeeze_dim_245 = torch.ops.aten.squeeze.dim(squeeze_dim_244, 0)
        squeeze_dim_244 = None
        unsqueeze_default_160 = torch.ops.aten.unsqueeze.default(primals_163, -1)
        unsqueeze_default_161 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_160, -1
        )
        unsqueeze_default_160 = None
        unsqueeze_default_162 = torch.ops.aten.unsqueeze.default(primals_164, -1)
        primals_164 = None
        unsqueeze_default_163 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_162, -1
        )
        unsqueeze_default_162 = None
        mul_tensor_286 = torch.ops.aten.mul.Tensor(
            mul_tensor_280, unsqueeze_default_161
        )
        mul_tensor_280 = unsqueeze_default_161 = None
        add_tensor_179 = torch.ops.aten.add.Tensor(
            mul_tensor_286, unsqueeze_default_163
        )
        mul_tensor_286 = unsqueeze_default_163 = None
        relu_default_40 = torch.ops.aten.relu.default(add_tensor_179)
        add_tensor_179 = None
        convolution_default_76 = torch.ops.aten.convolution.default(
            relu_default_40,
            primals_165,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_77 = torch.ops.aten.convolution.default(
            convolution_default_76,
            primals_166,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_82 = torch.ops.aten.var.correction(
            convolution_default_77, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_41 = torch.ops.aten.mean.dim(convolution_default_77, [0, 2, 3], True)
        add_tensor_180 = torch.ops.aten.add.Tensor(var_correction_82, 0.001)
        var_correction_82 = None
        sqrt_default_41 = torch.ops.aten.sqrt.default(add_tensor_180)
        add_tensor_180 = None
        reciprocal_default_41 = torch.ops.aten.reciprocal.default(sqrt_default_41)
        sqrt_default_41 = None
        sub_tensor_41 = torch.ops.aten.sub.Tensor(convolution_default_77, mean_dim_41)
        mul_tensor_287 = torch.ops.aten.mul.Tensor(sub_tensor_41, reciprocal_default_41)
        sub_tensor_41 = None
        squeeze_dim_246 = torch.ops.aten.squeeze.dim(mean_dim_41, 3)
        mean_dim_41 = None
        squeeze_dim_247 = torch.ops.aten.squeeze.dim(squeeze_dim_246, 2)
        squeeze_dim_246 = None
        squeeze_dim_248 = torch.ops.aten.squeeze.dim(squeeze_dim_247, 0)
        squeeze_dim_247 = None
        squeeze_dim_249 = torch.ops.aten.squeeze.dim(reciprocal_default_41, 3)
        reciprocal_default_41 = None
        squeeze_dim_250 = torch.ops.aten.squeeze.dim(squeeze_dim_249, 2)
        squeeze_dim_249 = None
        squeeze_dim_251 = torch.ops.aten.squeeze.dim(squeeze_dim_250, 0)
        squeeze_dim_250 = None
        unsqueeze_default_164 = torch.ops.aten.unsqueeze.default(primals_167, -1)
        unsqueeze_default_165 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_164, -1
        )
        unsqueeze_default_164 = None
        unsqueeze_default_166 = torch.ops.aten.unsqueeze.default(primals_168, -1)
        primals_168 = None
        unsqueeze_default_167 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_166, -1
        )
        unsqueeze_default_166 = None
        mul_tensor_293 = torch.ops.aten.mul.Tensor(
            mul_tensor_287, unsqueeze_default_165
        )
        mul_tensor_287 = unsqueeze_default_165 = None
        add_tensor_183 = torch.ops.aten.add.Tensor(
            mul_tensor_293, unsqueeze_default_167
        )
        mul_tensor_293 = unsqueeze_default_167 = None
        add_tensor_184 = torch.ops.aten.add.Tensor(add_tensor_175, add_tensor_183)
        add_tensor_175 = add_tensor_183 = None
        convolution_default_78 = torch.ops.aten.convolution.default(
            relu_default_39,
            primals_169,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_79 = torch.ops.aten.convolution.default(
            convolution_default_78,
            primals_170,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_84 = torch.ops.aten.var.correction(
            convolution_default_79, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_42 = torch.ops.aten.mean.dim(convolution_default_79, [0, 2, 3], True)
        add_tensor_185 = torch.ops.aten.add.Tensor(var_correction_84, 0.001)
        var_correction_84 = None
        sqrt_default_42 = torch.ops.aten.sqrt.default(add_tensor_185)
        add_tensor_185 = None
        reciprocal_default_42 = torch.ops.aten.reciprocal.default(sqrt_default_42)
        sqrt_default_42 = None
        sub_tensor_42 = torch.ops.aten.sub.Tensor(convolution_default_79, mean_dim_42)
        mul_tensor_294 = torch.ops.aten.mul.Tensor(sub_tensor_42, reciprocal_default_42)
        sub_tensor_42 = None
        squeeze_dim_252 = torch.ops.aten.squeeze.dim(mean_dim_42, 3)
        mean_dim_42 = None
        squeeze_dim_253 = torch.ops.aten.squeeze.dim(squeeze_dim_252, 2)
        squeeze_dim_252 = None
        squeeze_dim_254 = torch.ops.aten.squeeze.dim(squeeze_dim_253, 0)
        squeeze_dim_253 = None
        squeeze_dim_255 = torch.ops.aten.squeeze.dim(reciprocal_default_42, 3)
        reciprocal_default_42 = None
        squeeze_dim_256 = torch.ops.aten.squeeze.dim(squeeze_dim_255, 2)
        squeeze_dim_255 = None
        squeeze_dim_257 = torch.ops.aten.squeeze.dim(squeeze_dim_256, 0)
        squeeze_dim_256 = None
        unsqueeze_default_168 = torch.ops.aten.unsqueeze.default(primals_171, -1)
        unsqueeze_default_169 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_168, -1
        )
        unsqueeze_default_168 = None
        unsqueeze_default_170 = torch.ops.aten.unsqueeze.default(primals_172, -1)
        primals_172 = None
        unsqueeze_default_171 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_170, -1
        )
        unsqueeze_default_170 = None
        mul_tensor_300 = torch.ops.aten.mul.Tensor(
            mul_tensor_294, unsqueeze_default_169
        )
        mul_tensor_294 = unsqueeze_default_169 = None
        add_tensor_188 = torch.ops.aten.add.Tensor(
            mul_tensor_300, unsqueeze_default_171
        )
        mul_tensor_300 = unsqueeze_default_171 = None
        relu_default_42 = torch.ops.aten.relu.default(add_tensor_188)
        add_tensor_188 = None
        convolution_default_80 = torch.ops.aten.convolution.default(
            relu_default_42,
            primals_173,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_81 = torch.ops.aten.convolution.default(
            convolution_default_80,
            primals_174,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_86 = torch.ops.aten.var.correction(
            convolution_default_81, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_43 = torch.ops.aten.mean.dim(convolution_default_81, [0, 2, 3], True)
        add_tensor_189 = torch.ops.aten.add.Tensor(var_correction_86, 0.001)
        var_correction_86 = None
        sqrt_default_43 = torch.ops.aten.sqrt.default(add_tensor_189)
        add_tensor_189 = None
        reciprocal_default_43 = torch.ops.aten.reciprocal.default(sqrt_default_43)
        sqrt_default_43 = None
        sub_tensor_43 = torch.ops.aten.sub.Tensor(convolution_default_81, mean_dim_43)
        mul_tensor_301 = torch.ops.aten.mul.Tensor(sub_tensor_43, reciprocal_default_43)
        sub_tensor_43 = None
        squeeze_dim_258 = torch.ops.aten.squeeze.dim(mean_dim_43, 3)
        mean_dim_43 = None
        squeeze_dim_259 = torch.ops.aten.squeeze.dim(squeeze_dim_258, 2)
        squeeze_dim_258 = None
        squeeze_dim_260 = torch.ops.aten.squeeze.dim(squeeze_dim_259, 0)
        squeeze_dim_259 = None
        squeeze_dim_261 = torch.ops.aten.squeeze.dim(reciprocal_default_43, 3)
        reciprocal_default_43 = None
        squeeze_dim_262 = torch.ops.aten.squeeze.dim(squeeze_dim_261, 2)
        squeeze_dim_261 = None
        squeeze_dim_263 = torch.ops.aten.squeeze.dim(squeeze_dim_262, 0)
        squeeze_dim_262 = None
        unsqueeze_default_172 = torch.ops.aten.unsqueeze.default(primals_175, -1)
        unsqueeze_default_173 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_172, -1
        )
        unsqueeze_default_172 = None
        unsqueeze_default_174 = torch.ops.aten.unsqueeze.default(primals_176, -1)
        primals_176 = None
        unsqueeze_default_175 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_174, -1
        )
        unsqueeze_default_174 = None
        mul_tensor_307 = torch.ops.aten.mul.Tensor(
            mul_tensor_301, unsqueeze_default_173
        )
        mul_tensor_301 = unsqueeze_default_173 = None
        add_tensor_192 = torch.ops.aten.add.Tensor(
            mul_tensor_307, unsqueeze_default_175
        )
        mul_tensor_307 = unsqueeze_default_175 = None
        convolution_default_82 = torch.ops.aten.convolution.default(
            relu_default_39,
            primals_177,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_39 = None
        convolution_default_83 = torch.ops.aten.convolution.default(
            convolution_default_82,
            primals_178,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_88 = torch.ops.aten.var.correction(
            convolution_default_83, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_44 = torch.ops.aten.mean.dim(convolution_default_83, [0, 2, 3], True)
        add_tensor_193 = torch.ops.aten.add.Tensor(var_correction_88, 0.001)
        var_correction_88 = None
        sqrt_default_44 = torch.ops.aten.sqrt.default(add_tensor_193)
        add_tensor_193 = None
        reciprocal_default_44 = torch.ops.aten.reciprocal.default(sqrt_default_44)
        sqrt_default_44 = None
        sub_tensor_44 = torch.ops.aten.sub.Tensor(convolution_default_83, mean_dim_44)
        mul_tensor_308 = torch.ops.aten.mul.Tensor(sub_tensor_44, reciprocal_default_44)
        sub_tensor_44 = None
        squeeze_dim_264 = torch.ops.aten.squeeze.dim(mean_dim_44, 3)
        mean_dim_44 = None
        squeeze_dim_265 = torch.ops.aten.squeeze.dim(squeeze_dim_264, 2)
        squeeze_dim_264 = None
        squeeze_dim_266 = torch.ops.aten.squeeze.dim(squeeze_dim_265, 0)
        squeeze_dim_265 = None
        squeeze_dim_267 = torch.ops.aten.squeeze.dim(reciprocal_default_44, 3)
        reciprocal_default_44 = None
        squeeze_dim_268 = torch.ops.aten.squeeze.dim(squeeze_dim_267, 2)
        squeeze_dim_267 = None
        squeeze_dim_269 = torch.ops.aten.squeeze.dim(squeeze_dim_268, 0)
        squeeze_dim_268 = None
        unsqueeze_default_176 = torch.ops.aten.unsqueeze.default(primals_179, -1)
        unsqueeze_default_177 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_176, -1
        )
        unsqueeze_default_176 = None
        unsqueeze_default_178 = torch.ops.aten.unsqueeze.default(primals_180, -1)
        primals_180 = None
        unsqueeze_default_179 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_178, -1
        )
        unsqueeze_default_178 = None
        mul_tensor_314 = torch.ops.aten.mul.Tensor(
            mul_tensor_308, unsqueeze_default_177
        )
        mul_tensor_308 = unsqueeze_default_177 = None
        add_tensor_196 = torch.ops.aten.add.Tensor(
            mul_tensor_314, unsqueeze_default_179
        )
        mul_tensor_314 = unsqueeze_default_179 = None
        relu_default_44 = torch.ops.aten.relu.default(add_tensor_196)
        add_tensor_196 = None
        convolution_default_84 = torch.ops.aten.convolution.default(
            relu_default_44,
            primals_181,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_85 = torch.ops.aten.convolution.default(
            convolution_default_84,
            primals_182,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_90 = torch.ops.aten.var.correction(
            convolution_default_85, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_45 = torch.ops.aten.mean.dim(convolution_default_85, [0, 2, 3], True)
        add_tensor_197 = torch.ops.aten.add.Tensor(var_correction_90, 0.001)
        var_correction_90 = None
        sqrt_default_45 = torch.ops.aten.sqrt.default(add_tensor_197)
        add_tensor_197 = None
        reciprocal_default_45 = torch.ops.aten.reciprocal.default(sqrt_default_45)
        sqrt_default_45 = None
        sub_tensor_45 = torch.ops.aten.sub.Tensor(convolution_default_85, mean_dim_45)
        mul_tensor_315 = torch.ops.aten.mul.Tensor(sub_tensor_45, reciprocal_default_45)
        sub_tensor_45 = None
        squeeze_dim_270 = torch.ops.aten.squeeze.dim(mean_dim_45, 3)
        mean_dim_45 = None
        squeeze_dim_271 = torch.ops.aten.squeeze.dim(squeeze_dim_270, 2)
        squeeze_dim_270 = None
        squeeze_dim_272 = torch.ops.aten.squeeze.dim(squeeze_dim_271, 0)
        squeeze_dim_271 = None
        squeeze_dim_273 = torch.ops.aten.squeeze.dim(reciprocal_default_45, 3)
        reciprocal_default_45 = None
        squeeze_dim_274 = torch.ops.aten.squeeze.dim(squeeze_dim_273, 2)
        squeeze_dim_273 = None
        squeeze_dim_275 = torch.ops.aten.squeeze.dim(squeeze_dim_274, 0)
        squeeze_dim_274 = None
        unsqueeze_default_180 = torch.ops.aten.unsqueeze.default(primals_183, -1)
        unsqueeze_default_181 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_180, -1
        )
        unsqueeze_default_180 = None
        unsqueeze_default_182 = torch.ops.aten.unsqueeze.default(primals_184, -1)
        primals_184 = None
        unsqueeze_default_183 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_182, -1
        )
        unsqueeze_default_182 = None
        mul_tensor_321 = torch.ops.aten.mul.Tensor(
            mul_tensor_315, unsqueeze_default_181
        )
        mul_tensor_315 = unsqueeze_default_181 = None
        add_tensor_200 = torch.ops.aten.add.Tensor(
            mul_tensor_321, unsqueeze_default_183
        )
        mul_tensor_321 = unsqueeze_default_183 = None
        add_tensor_201 = torch.ops.aten.add.Tensor(add_tensor_192, add_tensor_200)
        add_tensor_192 = add_tensor_200 = None
        avg_pool2d_default_11 = torch.ops.aten.avg_pool2d.default(
            add_tensor_167, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_202 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_11, add_tensor_163
        )
        avg_pool2d_default_11 = None
        avg_pool2d_default_12 = torch.ops.aten.avg_pool2d.default(
            add_tensor_163, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_203 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_12, avg_pool2d_default_12
        )
        avg_pool2d_default_12 = None
        convolution_default_86 = torch.ops.aten.convolution.default(
            relu_default_37,
            primals_185,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_37 = None
        convolution_default_87 = torch.ops.aten.convolution.default(
            convolution_default_86,
            primals_186,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_92 = torch.ops.aten.var.correction(
            convolution_default_87, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_46 = torch.ops.aten.mean.dim(convolution_default_87, [0, 2, 3], True)
        add_tensor_204 = torch.ops.aten.add.Tensor(var_correction_92, 0.001)
        var_correction_92 = None
        sqrt_default_46 = torch.ops.aten.sqrt.default(add_tensor_204)
        add_tensor_204 = None
        reciprocal_default_46 = torch.ops.aten.reciprocal.default(sqrt_default_46)
        sqrt_default_46 = None
        sub_tensor_46 = torch.ops.aten.sub.Tensor(convolution_default_87, mean_dim_46)
        mul_tensor_322 = torch.ops.aten.mul.Tensor(sub_tensor_46, reciprocal_default_46)
        sub_tensor_46 = None
        squeeze_dim_276 = torch.ops.aten.squeeze.dim(mean_dim_46, 3)
        mean_dim_46 = None
        squeeze_dim_277 = torch.ops.aten.squeeze.dim(squeeze_dim_276, 2)
        squeeze_dim_276 = None
        squeeze_dim_278 = torch.ops.aten.squeeze.dim(squeeze_dim_277, 0)
        squeeze_dim_277 = None
        squeeze_dim_279 = torch.ops.aten.squeeze.dim(reciprocal_default_46, 3)
        reciprocal_default_46 = None
        squeeze_dim_280 = torch.ops.aten.squeeze.dim(squeeze_dim_279, 2)
        squeeze_dim_279 = None
        squeeze_dim_281 = torch.ops.aten.squeeze.dim(squeeze_dim_280, 0)
        squeeze_dim_280 = None
        unsqueeze_default_184 = torch.ops.aten.unsqueeze.default(primals_187, -1)
        unsqueeze_default_185 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_184, -1
        )
        unsqueeze_default_184 = None
        unsqueeze_default_186 = torch.ops.aten.unsqueeze.default(primals_188, -1)
        primals_188 = None
        unsqueeze_default_187 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_186, -1
        )
        unsqueeze_default_186 = None
        mul_tensor_328 = torch.ops.aten.mul.Tensor(
            mul_tensor_322, unsqueeze_default_185
        )
        mul_tensor_322 = unsqueeze_default_185 = None
        add_tensor_207 = torch.ops.aten.add.Tensor(
            mul_tensor_328, unsqueeze_default_187
        )
        mul_tensor_328 = unsqueeze_default_187 = None
        relu_default_46 = torch.ops.aten.relu.default(add_tensor_207)
        add_tensor_207 = None
        convolution_default_88 = torch.ops.aten.convolution.default(
            relu_default_46,
            primals_189,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_89 = torch.ops.aten.convolution.default(
            convolution_default_88,
            primals_190,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_94 = torch.ops.aten.var.correction(
            convolution_default_89, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_47 = torch.ops.aten.mean.dim(convolution_default_89, [0, 2, 3], True)
        add_tensor_208 = torch.ops.aten.add.Tensor(var_correction_94, 0.001)
        var_correction_94 = None
        sqrt_default_47 = torch.ops.aten.sqrt.default(add_tensor_208)
        add_tensor_208 = None
        reciprocal_default_47 = torch.ops.aten.reciprocal.default(sqrt_default_47)
        sqrt_default_47 = None
        sub_tensor_47 = torch.ops.aten.sub.Tensor(convolution_default_89, mean_dim_47)
        mul_tensor_329 = torch.ops.aten.mul.Tensor(sub_tensor_47, reciprocal_default_47)
        sub_tensor_47 = None
        squeeze_dim_282 = torch.ops.aten.squeeze.dim(mean_dim_47, 3)
        mean_dim_47 = None
        squeeze_dim_283 = torch.ops.aten.squeeze.dim(squeeze_dim_282, 2)
        squeeze_dim_282 = None
        squeeze_dim_284 = torch.ops.aten.squeeze.dim(squeeze_dim_283, 0)
        squeeze_dim_283 = None
        squeeze_dim_285 = torch.ops.aten.squeeze.dim(reciprocal_default_47, 3)
        reciprocal_default_47 = None
        squeeze_dim_286 = torch.ops.aten.squeeze.dim(squeeze_dim_285, 2)
        squeeze_dim_285 = None
        squeeze_dim_287 = torch.ops.aten.squeeze.dim(squeeze_dim_286, 0)
        squeeze_dim_286 = None
        unsqueeze_default_188 = torch.ops.aten.unsqueeze.default(primals_191, -1)
        unsqueeze_default_189 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_188, -1
        )
        unsqueeze_default_188 = None
        unsqueeze_default_190 = torch.ops.aten.unsqueeze.default(primals_192, -1)
        primals_192 = None
        unsqueeze_default_191 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_190, -1
        )
        unsqueeze_default_190 = None
        mul_tensor_335 = torch.ops.aten.mul.Tensor(
            mul_tensor_329, unsqueeze_default_189
        )
        mul_tensor_329 = unsqueeze_default_189 = None
        add_tensor_211 = torch.ops.aten.add.Tensor(
            mul_tensor_335, unsqueeze_default_191
        )
        mul_tensor_335 = unsqueeze_default_191 = None
        add_tensor_212 = torch.ops.aten.add.Tensor(add_tensor_211, add_tensor_167)
        add_tensor_211 = add_tensor_167 = None
        cat_default_5 = torch.ops.aten.cat.default(
            [
                add_tensor_163,
                add_tensor_184,
                add_tensor_201,
                add_tensor_202,
                add_tensor_203,
                add_tensor_212,
            ],
            1,
        )
        add_tensor_163 = (
            add_tensor_184
        ) = add_tensor_201 = add_tensor_202 = add_tensor_203 = add_tensor_212 = None
        convolution_default_90 = torch.ops.aten.convolution.default(
            relu_default_36, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_96 = torch.ops.aten.var.correction(
            convolution_default_90, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_48 = torch.ops.aten.mean.dim(convolution_default_90, [0, 2, 3], True)
        add_tensor_213 = torch.ops.aten.add.Tensor(var_correction_96, 0.001)
        var_correction_96 = None
        sqrt_default_48 = torch.ops.aten.sqrt.default(add_tensor_213)
        add_tensor_213 = None
        reciprocal_default_48 = torch.ops.aten.reciprocal.default(sqrt_default_48)
        sqrt_default_48 = None
        sub_tensor_48 = torch.ops.aten.sub.Tensor(convolution_default_90, mean_dim_48)
        mul_tensor_336 = torch.ops.aten.mul.Tensor(sub_tensor_48, reciprocal_default_48)
        sub_tensor_48 = None
        unsqueeze_default_192 = torch.ops.aten.unsqueeze.default(primals_194, -1)
        unsqueeze_default_193 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_192, -1
        )
        unsqueeze_default_192 = None
        unsqueeze_default_194 = torch.ops.aten.unsqueeze.default(primals_195, -1)
        unsqueeze_default_195 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_194, -1
        )
        unsqueeze_default_194 = None
        mul_tensor_342 = torch.ops.aten.mul.Tensor(
            mul_tensor_336, unsqueeze_default_193
        )
        mul_tensor_336 = unsqueeze_default_193 = None
        add_tensor_216 = torch.ops.aten.add.Tensor(
            mul_tensor_342, unsqueeze_default_195
        )
        mul_tensor_342 = unsqueeze_default_195 = None
        relu_default_48 = torch.ops.aten.relu.default(cat_default_5)
        cat_default_5 = None
        convolution_default_91 = torch.ops.aten.convolution.default(
            relu_default_48, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_98 = torch.ops.aten.var.correction(
            convolution_default_91, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_49 = torch.ops.aten.mean.dim(convolution_default_91, [0, 2, 3], True)
        add_tensor_217 = torch.ops.aten.add.Tensor(var_correction_98, 0.001)
        var_correction_98 = None
        sqrt_default_49 = torch.ops.aten.sqrt.default(add_tensor_217)
        add_tensor_217 = None
        reciprocal_default_49 = torch.ops.aten.reciprocal.default(sqrt_default_49)
        sqrt_default_49 = None
        sub_tensor_49 = torch.ops.aten.sub.Tensor(convolution_default_91, mean_dim_49)
        mul_tensor_343 = torch.ops.aten.mul.Tensor(sub_tensor_49, reciprocal_default_49)
        sub_tensor_49 = None
        unsqueeze_default_196 = torch.ops.aten.unsqueeze.default(primals_197, -1)
        unsqueeze_default_197 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_196, -1
        )
        unsqueeze_default_196 = None
        unsqueeze_default_198 = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_default_199 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_198, -1
        )
        unsqueeze_default_198 = None
        mul_tensor_349 = torch.ops.aten.mul.Tensor(
            mul_tensor_343, unsqueeze_default_197
        )
        mul_tensor_343 = unsqueeze_default_197 = None
        add_tensor_220 = torch.ops.aten.add.Tensor(
            mul_tensor_349, unsqueeze_default_199
        )
        mul_tensor_349 = unsqueeze_default_199 = None
        relu_default_49 = torch.ops.aten.relu.default(add_tensor_220)
        convolution_default_92 = torch.ops.aten.convolution.default(
            relu_default_49,
            primals_199,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_93 = torch.ops.aten.convolution.default(
            convolution_default_92,
            primals_200,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_100 = torch.ops.aten.var.correction(
            convolution_default_93, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_50 = torch.ops.aten.mean.dim(convolution_default_93, [0, 2, 3], True)
        add_tensor_221 = torch.ops.aten.add.Tensor(var_correction_100, 0.001)
        var_correction_100 = None
        sqrt_default_50 = torch.ops.aten.sqrt.default(add_tensor_221)
        add_tensor_221 = None
        reciprocal_default_50 = torch.ops.aten.reciprocal.default(sqrt_default_50)
        sqrt_default_50 = None
        sub_tensor_50 = torch.ops.aten.sub.Tensor(convolution_default_93, mean_dim_50)
        mul_tensor_350 = torch.ops.aten.mul.Tensor(sub_tensor_50, reciprocal_default_50)
        sub_tensor_50 = None
        squeeze_dim_300 = torch.ops.aten.squeeze.dim(mean_dim_50, 3)
        mean_dim_50 = None
        squeeze_dim_301 = torch.ops.aten.squeeze.dim(squeeze_dim_300, 2)
        squeeze_dim_300 = None
        squeeze_dim_302 = torch.ops.aten.squeeze.dim(squeeze_dim_301, 0)
        squeeze_dim_301 = None
        squeeze_dim_303 = torch.ops.aten.squeeze.dim(reciprocal_default_50, 3)
        reciprocal_default_50 = None
        squeeze_dim_304 = torch.ops.aten.squeeze.dim(squeeze_dim_303, 2)
        squeeze_dim_303 = None
        squeeze_dim_305 = torch.ops.aten.squeeze.dim(squeeze_dim_304, 0)
        squeeze_dim_304 = None
        unsqueeze_default_200 = torch.ops.aten.unsqueeze.default(primals_201, -1)
        unsqueeze_default_201 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_200, -1
        )
        unsqueeze_default_200 = None
        unsqueeze_default_202 = torch.ops.aten.unsqueeze.default(primals_202, -1)
        primals_202 = None
        unsqueeze_default_203 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_202, -1
        )
        unsqueeze_default_202 = None
        mul_tensor_356 = torch.ops.aten.mul.Tensor(
            mul_tensor_350, unsqueeze_default_201
        )
        mul_tensor_350 = unsqueeze_default_201 = None
        add_tensor_224 = torch.ops.aten.add.Tensor(
            mul_tensor_356, unsqueeze_default_203
        )
        mul_tensor_356 = unsqueeze_default_203 = None
        relu_default_50 = torch.ops.aten.relu.default(add_tensor_224)
        add_tensor_224 = None
        convolution_default_94 = torch.ops.aten.convolution.default(
            relu_default_50,
            primals_203,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_95 = torch.ops.aten.convolution.default(
            convolution_default_94,
            primals_204,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_102 = torch.ops.aten.var.correction(
            convolution_default_95, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_51 = torch.ops.aten.mean.dim(convolution_default_95, [0, 2, 3], True)
        add_tensor_225 = torch.ops.aten.add.Tensor(var_correction_102, 0.001)
        var_correction_102 = None
        sqrt_default_51 = torch.ops.aten.sqrt.default(add_tensor_225)
        add_tensor_225 = None
        reciprocal_default_51 = torch.ops.aten.reciprocal.default(sqrt_default_51)
        sqrt_default_51 = None
        sub_tensor_51 = torch.ops.aten.sub.Tensor(convolution_default_95, mean_dim_51)
        mul_tensor_357 = torch.ops.aten.mul.Tensor(sub_tensor_51, reciprocal_default_51)
        sub_tensor_51 = None
        squeeze_dim_306 = torch.ops.aten.squeeze.dim(mean_dim_51, 3)
        mean_dim_51 = None
        squeeze_dim_307 = torch.ops.aten.squeeze.dim(squeeze_dim_306, 2)
        squeeze_dim_306 = None
        squeeze_dim_308 = torch.ops.aten.squeeze.dim(squeeze_dim_307, 0)
        squeeze_dim_307 = None
        squeeze_dim_309 = torch.ops.aten.squeeze.dim(reciprocal_default_51, 3)
        reciprocal_default_51 = None
        squeeze_dim_310 = torch.ops.aten.squeeze.dim(squeeze_dim_309, 2)
        squeeze_dim_309 = None
        squeeze_dim_311 = torch.ops.aten.squeeze.dim(squeeze_dim_310, 0)
        squeeze_dim_310 = None
        unsqueeze_default_204 = torch.ops.aten.unsqueeze.default(primals_205, -1)
        unsqueeze_default_205 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_204, -1
        )
        unsqueeze_default_204 = None
        unsqueeze_default_206 = torch.ops.aten.unsqueeze.default(primals_206, -1)
        primals_206 = None
        unsqueeze_default_207 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_206, -1
        )
        unsqueeze_default_206 = None
        mul_tensor_363 = torch.ops.aten.mul.Tensor(
            mul_tensor_357, unsqueeze_default_205
        )
        mul_tensor_357 = unsqueeze_default_205 = None
        add_tensor_228 = torch.ops.aten.add.Tensor(
            mul_tensor_363, unsqueeze_default_207
        )
        mul_tensor_363 = unsqueeze_default_207 = None
        relu_default_51 = torch.ops.aten.relu.default(add_tensor_216)
        convolution_default_96 = torch.ops.aten.convolution.default(
            relu_default_51,
            primals_207,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_97 = torch.ops.aten.convolution.default(
            convolution_default_96,
            primals_208,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_104 = torch.ops.aten.var.correction(
            convolution_default_97, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_52 = torch.ops.aten.mean.dim(convolution_default_97, [0, 2, 3], True)
        add_tensor_229 = torch.ops.aten.add.Tensor(var_correction_104, 0.001)
        var_correction_104 = None
        sqrt_default_52 = torch.ops.aten.sqrt.default(add_tensor_229)
        add_tensor_229 = None
        reciprocal_default_52 = torch.ops.aten.reciprocal.default(sqrt_default_52)
        sqrt_default_52 = None
        sub_tensor_52 = torch.ops.aten.sub.Tensor(convolution_default_97, mean_dim_52)
        mul_tensor_364 = torch.ops.aten.mul.Tensor(sub_tensor_52, reciprocal_default_52)
        sub_tensor_52 = None
        squeeze_dim_312 = torch.ops.aten.squeeze.dim(mean_dim_52, 3)
        mean_dim_52 = None
        squeeze_dim_313 = torch.ops.aten.squeeze.dim(squeeze_dim_312, 2)
        squeeze_dim_312 = None
        squeeze_dim_314 = torch.ops.aten.squeeze.dim(squeeze_dim_313, 0)
        squeeze_dim_313 = None
        squeeze_dim_315 = torch.ops.aten.squeeze.dim(reciprocal_default_52, 3)
        reciprocal_default_52 = None
        squeeze_dim_316 = torch.ops.aten.squeeze.dim(squeeze_dim_315, 2)
        squeeze_dim_315 = None
        squeeze_dim_317 = torch.ops.aten.squeeze.dim(squeeze_dim_316, 0)
        squeeze_dim_316 = None
        unsqueeze_default_208 = torch.ops.aten.unsqueeze.default(primals_209, -1)
        unsqueeze_default_209 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_208, -1
        )
        unsqueeze_default_208 = None
        unsqueeze_default_210 = torch.ops.aten.unsqueeze.default(primals_210, -1)
        primals_210 = None
        unsqueeze_default_211 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_210, -1
        )
        unsqueeze_default_210 = None
        mul_tensor_370 = torch.ops.aten.mul.Tensor(
            mul_tensor_364, unsqueeze_default_209
        )
        mul_tensor_364 = unsqueeze_default_209 = None
        add_tensor_232 = torch.ops.aten.add.Tensor(
            mul_tensor_370, unsqueeze_default_211
        )
        mul_tensor_370 = unsqueeze_default_211 = None
        relu_default_52 = torch.ops.aten.relu.default(add_tensor_232)
        add_tensor_232 = None
        convolution_default_98 = torch.ops.aten.convolution.default(
            relu_default_52,
            primals_211,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_99 = torch.ops.aten.convolution.default(
            convolution_default_98,
            primals_212,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_106 = torch.ops.aten.var.correction(
            convolution_default_99, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_53 = torch.ops.aten.mean.dim(convolution_default_99, [0, 2, 3], True)
        add_tensor_233 = torch.ops.aten.add.Tensor(var_correction_106, 0.001)
        var_correction_106 = None
        sqrt_default_53 = torch.ops.aten.sqrt.default(add_tensor_233)
        add_tensor_233 = None
        reciprocal_default_53 = torch.ops.aten.reciprocal.default(sqrt_default_53)
        sqrt_default_53 = None
        sub_tensor_53 = torch.ops.aten.sub.Tensor(convolution_default_99, mean_dim_53)
        mul_tensor_371 = torch.ops.aten.mul.Tensor(sub_tensor_53, reciprocal_default_53)
        sub_tensor_53 = None
        squeeze_dim_318 = torch.ops.aten.squeeze.dim(mean_dim_53, 3)
        mean_dim_53 = None
        squeeze_dim_319 = torch.ops.aten.squeeze.dim(squeeze_dim_318, 2)
        squeeze_dim_318 = None
        squeeze_dim_320 = torch.ops.aten.squeeze.dim(squeeze_dim_319, 0)
        squeeze_dim_319 = None
        squeeze_dim_321 = torch.ops.aten.squeeze.dim(reciprocal_default_53, 3)
        reciprocal_default_53 = None
        squeeze_dim_322 = torch.ops.aten.squeeze.dim(squeeze_dim_321, 2)
        squeeze_dim_321 = None
        squeeze_dim_323 = torch.ops.aten.squeeze.dim(squeeze_dim_322, 0)
        squeeze_dim_322 = None
        unsqueeze_default_212 = torch.ops.aten.unsqueeze.default(primals_213, -1)
        unsqueeze_default_213 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_212, -1
        )
        unsqueeze_default_212 = None
        unsqueeze_default_214 = torch.ops.aten.unsqueeze.default(primals_214, -1)
        primals_214 = None
        unsqueeze_default_215 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_214, -1
        )
        unsqueeze_default_214 = None
        mul_tensor_377 = torch.ops.aten.mul.Tensor(
            mul_tensor_371, unsqueeze_default_213
        )
        mul_tensor_371 = unsqueeze_default_213 = None
        add_tensor_236 = torch.ops.aten.add.Tensor(
            mul_tensor_377, unsqueeze_default_215
        )
        mul_tensor_377 = unsqueeze_default_215 = None
        add_tensor_237 = torch.ops.aten.add.Tensor(add_tensor_228, add_tensor_236)
        add_tensor_228 = add_tensor_236 = None
        convolution_default_100 = torch.ops.aten.convolution.default(
            relu_default_51,
            primals_215,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_101 = torch.ops.aten.convolution.default(
            convolution_default_100,
            primals_216,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_108 = torch.ops.aten.var.correction(
            convolution_default_101, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_54 = torch.ops.aten.mean.dim(convolution_default_101, [0, 2, 3], True)
        add_tensor_238 = torch.ops.aten.add.Tensor(var_correction_108, 0.001)
        var_correction_108 = None
        sqrt_default_54 = torch.ops.aten.sqrt.default(add_tensor_238)
        add_tensor_238 = None
        reciprocal_default_54 = torch.ops.aten.reciprocal.default(sqrt_default_54)
        sqrt_default_54 = None
        sub_tensor_54 = torch.ops.aten.sub.Tensor(convolution_default_101, mean_dim_54)
        mul_tensor_378 = torch.ops.aten.mul.Tensor(sub_tensor_54, reciprocal_default_54)
        sub_tensor_54 = None
        squeeze_dim_324 = torch.ops.aten.squeeze.dim(mean_dim_54, 3)
        mean_dim_54 = None
        squeeze_dim_325 = torch.ops.aten.squeeze.dim(squeeze_dim_324, 2)
        squeeze_dim_324 = None
        squeeze_dim_326 = torch.ops.aten.squeeze.dim(squeeze_dim_325, 0)
        squeeze_dim_325 = None
        squeeze_dim_327 = torch.ops.aten.squeeze.dim(reciprocal_default_54, 3)
        reciprocal_default_54 = None
        squeeze_dim_328 = torch.ops.aten.squeeze.dim(squeeze_dim_327, 2)
        squeeze_dim_327 = None
        squeeze_dim_329 = torch.ops.aten.squeeze.dim(squeeze_dim_328, 0)
        squeeze_dim_328 = None
        unsqueeze_default_216 = torch.ops.aten.unsqueeze.default(primals_217, -1)
        unsqueeze_default_217 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_216, -1
        )
        unsqueeze_default_216 = None
        unsqueeze_default_218 = torch.ops.aten.unsqueeze.default(primals_218, -1)
        primals_218 = None
        unsqueeze_default_219 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_218, -1
        )
        unsqueeze_default_218 = None
        mul_tensor_384 = torch.ops.aten.mul.Tensor(
            mul_tensor_378, unsqueeze_default_217
        )
        mul_tensor_378 = unsqueeze_default_217 = None
        add_tensor_241 = torch.ops.aten.add.Tensor(
            mul_tensor_384, unsqueeze_default_219
        )
        mul_tensor_384 = unsqueeze_default_219 = None
        relu_default_54 = torch.ops.aten.relu.default(add_tensor_241)
        add_tensor_241 = None
        convolution_default_102 = torch.ops.aten.convolution.default(
            relu_default_54,
            primals_219,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_103 = torch.ops.aten.convolution.default(
            convolution_default_102,
            primals_220,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_110 = torch.ops.aten.var.correction(
            convolution_default_103, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_55 = torch.ops.aten.mean.dim(convolution_default_103, [0, 2, 3], True)
        add_tensor_242 = torch.ops.aten.add.Tensor(var_correction_110, 0.001)
        var_correction_110 = None
        sqrt_default_55 = torch.ops.aten.sqrt.default(add_tensor_242)
        add_tensor_242 = None
        reciprocal_default_55 = torch.ops.aten.reciprocal.default(sqrt_default_55)
        sqrt_default_55 = None
        sub_tensor_55 = torch.ops.aten.sub.Tensor(convolution_default_103, mean_dim_55)
        mul_tensor_385 = torch.ops.aten.mul.Tensor(sub_tensor_55, reciprocal_default_55)
        sub_tensor_55 = None
        squeeze_dim_330 = torch.ops.aten.squeeze.dim(mean_dim_55, 3)
        mean_dim_55 = None
        squeeze_dim_331 = torch.ops.aten.squeeze.dim(squeeze_dim_330, 2)
        squeeze_dim_330 = None
        squeeze_dim_332 = torch.ops.aten.squeeze.dim(squeeze_dim_331, 0)
        squeeze_dim_331 = None
        squeeze_dim_333 = torch.ops.aten.squeeze.dim(reciprocal_default_55, 3)
        reciprocal_default_55 = None
        squeeze_dim_334 = torch.ops.aten.squeeze.dim(squeeze_dim_333, 2)
        squeeze_dim_333 = None
        squeeze_dim_335 = torch.ops.aten.squeeze.dim(squeeze_dim_334, 0)
        squeeze_dim_334 = None
        unsqueeze_default_220 = torch.ops.aten.unsqueeze.default(primals_221, -1)
        unsqueeze_default_221 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_220, -1
        )
        unsqueeze_default_220 = None
        unsqueeze_default_222 = torch.ops.aten.unsqueeze.default(primals_222, -1)
        primals_222 = None
        unsqueeze_default_223 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_222, -1
        )
        unsqueeze_default_222 = None
        mul_tensor_391 = torch.ops.aten.mul.Tensor(
            mul_tensor_385, unsqueeze_default_221
        )
        mul_tensor_385 = unsqueeze_default_221 = None
        add_tensor_245 = torch.ops.aten.add.Tensor(
            mul_tensor_391, unsqueeze_default_223
        )
        mul_tensor_391 = unsqueeze_default_223 = None
        convolution_default_104 = torch.ops.aten.convolution.default(
            relu_default_51,
            primals_223,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_51 = None
        convolution_default_105 = torch.ops.aten.convolution.default(
            convolution_default_104,
            primals_224,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_112 = torch.ops.aten.var.correction(
            convolution_default_105, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_56 = torch.ops.aten.mean.dim(convolution_default_105, [0, 2, 3], True)
        add_tensor_246 = torch.ops.aten.add.Tensor(var_correction_112, 0.001)
        var_correction_112 = None
        sqrt_default_56 = torch.ops.aten.sqrt.default(add_tensor_246)
        add_tensor_246 = None
        reciprocal_default_56 = torch.ops.aten.reciprocal.default(sqrt_default_56)
        sqrt_default_56 = None
        sub_tensor_56 = torch.ops.aten.sub.Tensor(convolution_default_105, mean_dim_56)
        mul_tensor_392 = torch.ops.aten.mul.Tensor(sub_tensor_56, reciprocal_default_56)
        sub_tensor_56 = None
        squeeze_dim_336 = torch.ops.aten.squeeze.dim(mean_dim_56, 3)
        mean_dim_56 = None
        squeeze_dim_337 = torch.ops.aten.squeeze.dim(squeeze_dim_336, 2)
        squeeze_dim_336 = None
        squeeze_dim_338 = torch.ops.aten.squeeze.dim(squeeze_dim_337, 0)
        squeeze_dim_337 = None
        squeeze_dim_339 = torch.ops.aten.squeeze.dim(reciprocal_default_56, 3)
        reciprocal_default_56 = None
        squeeze_dim_340 = torch.ops.aten.squeeze.dim(squeeze_dim_339, 2)
        squeeze_dim_339 = None
        squeeze_dim_341 = torch.ops.aten.squeeze.dim(squeeze_dim_340, 0)
        squeeze_dim_340 = None
        unsqueeze_default_224 = torch.ops.aten.unsqueeze.default(primals_225, -1)
        unsqueeze_default_225 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_224, -1
        )
        unsqueeze_default_224 = None
        unsqueeze_default_226 = torch.ops.aten.unsqueeze.default(primals_226, -1)
        primals_226 = None
        unsqueeze_default_227 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_226, -1
        )
        unsqueeze_default_226 = None
        mul_tensor_398 = torch.ops.aten.mul.Tensor(
            mul_tensor_392, unsqueeze_default_225
        )
        mul_tensor_392 = unsqueeze_default_225 = None
        add_tensor_249 = torch.ops.aten.add.Tensor(
            mul_tensor_398, unsqueeze_default_227
        )
        mul_tensor_398 = unsqueeze_default_227 = None
        relu_default_56 = torch.ops.aten.relu.default(add_tensor_249)
        add_tensor_249 = None
        convolution_default_106 = torch.ops.aten.convolution.default(
            relu_default_56,
            primals_227,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_107 = torch.ops.aten.convolution.default(
            convolution_default_106,
            primals_228,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_114 = torch.ops.aten.var.correction(
            convolution_default_107, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_57 = torch.ops.aten.mean.dim(convolution_default_107, [0, 2, 3], True)
        add_tensor_250 = torch.ops.aten.add.Tensor(var_correction_114, 0.001)
        var_correction_114 = None
        sqrt_default_57 = torch.ops.aten.sqrt.default(add_tensor_250)
        add_tensor_250 = None
        reciprocal_default_57 = torch.ops.aten.reciprocal.default(sqrt_default_57)
        sqrt_default_57 = None
        sub_tensor_57 = torch.ops.aten.sub.Tensor(convolution_default_107, mean_dim_57)
        mul_tensor_399 = torch.ops.aten.mul.Tensor(sub_tensor_57, reciprocal_default_57)
        sub_tensor_57 = None
        squeeze_dim_342 = torch.ops.aten.squeeze.dim(mean_dim_57, 3)
        mean_dim_57 = None
        squeeze_dim_343 = torch.ops.aten.squeeze.dim(squeeze_dim_342, 2)
        squeeze_dim_342 = None
        squeeze_dim_344 = torch.ops.aten.squeeze.dim(squeeze_dim_343, 0)
        squeeze_dim_343 = None
        squeeze_dim_345 = torch.ops.aten.squeeze.dim(reciprocal_default_57, 3)
        reciprocal_default_57 = None
        squeeze_dim_346 = torch.ops.aten.squeeze.dim(squeeze_dim_345, 2)
        squeeze_dim_345 = None
        squeeze_dim_347 = torch.ops.aten.squeeze.dim(squeeze_dim_346, 0)
        squeeze_dim_346 = None
        unsqueeze_default_228 = torch.ops.aten.unsqueeze.default(primals_229, -1)
        unsqueeze_default_229 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_228, -1
        )
        unsqueeze_default_228 = None
        unsqueeze_default_230 = torch.ops.aten.unsqueeze.default(primals_230, -1)
        primals_230 = None
        unsqueeze_default_231 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_230, -1
        )
        unsqueeze_default_230 = None
        mul_tensor_405 = torch.ops.aten.mul.Tensor(
            mul_tensor_399, unsqueeze_default_229
        )
        mul_tensor_399 = unsqueeze_default_229 = None
        add_tensor_253 = torch.ops.aten.add.Tensor(
            mul_tensor_405, unsqueeze_default_231
        )
        mul_tensor_405 = unsqueeze_default_231 = None
        add_tensor_254 = torch.ops.aten.add.Tensor(add_tensor_245, add_tensor_253)
        add_tensor_245 = add_tensor_253 = None
        avg_pool2d_default_14 = torch.ops.aten.avg_pool2d.default(
            add_tensor_220, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_255 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_14, add_tensor_216
        )
        avg_pool2d_default_14 = None
        avg_pool2d_default_15 = torch.ops.aten.avg_pool2d.default(
            add_tensor_216, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_256 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_15, avg_pool2d_default_15
        )
        avg_pool2d_default_15 = None
        convolution_default_108 = torch.ops.aten.convolution.default(
            relu_default_49,
            primals_231,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_49 = None
        convolution_default_109 = torch.ops.aten.convolution.default(
            convolution_default_108,
            primals_232,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_116 = torch.ops.aten.var.correction(
            convolution_default_109, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_58 = torch.ops.aten.mean.dim(convolution_default_109, [0, 2, 3], True)
        add_tensor_257 = torch.ops.aten.add.Tensor(var_correction_116, 0.001)
        var_correction_116 = None
        sqrt_default_58 = torch.ops.aten.sqrt.default(add_tensor_257)
        add_tensor_257 = None
        reciprocal_default_58 = torch.ops.aten.reciprocal.default(sqrt_default_58)
        sqrt_default_58 = None
        sub_tensor_58 = torch.ops.aten.sub.Tensor(convolution_default_109, mean_dim_58)
        mul_tensor_406 = torch.ops.aten.mul.Tensor(sub_tensor_58, reciprocal_default_58)
        sub_tensor_58 = None
        squeeze_dim_348 = torch.ops.aten.squeeze.dim(mean_dim_58, 3)
        mean_dim_58 = None
        squeeze_dim_349 = torch.ops.aten.squeeze.dim(squeeze_dim_348, 2)
        squeeze_dim_348 = None
        squeeze_dim_350 = torch.ops.aten.squeeze.dim(squeeze_dim_349, 0)
        squeeze_dim_349 = None
        squeeze_dim_351 = torch.ops.aten.squeeze.dim(reciprocal_default_58, 3)
        reciprocal_default_58 = None
        squeeze_dim_352 = torch.ops.aten.squeeze.dim(squeeze_dim_351, 2)
        squeeze_dim_351 = None
        squeeze_dim_353 = torch.ops.aten.squeeze.dim(squeeze_dim_352, 0)
        squeeze_dim_352 = None
        unsqueeze_default_232 = torch.ops.aten.unsqueeze.default(primals_233, -1)
        unsqueeze_default_233 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_232, -1
        )
        unsqueeze_default_232 = None
        unsqueeze_default_234 = torch.ops.aten.unsqueeze.default(primals_234, -1)
        primals_234 = None
        unsqueeze_default_235 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_234, -1
        )
        unsqueeze_default_234 = None
        mul_tensor_412 = torch.ops.aten.mul.Tensor(
            mul_tensor_406, unsqueeze_default_233
        )
        mul_tensor_406 = unsqueeze_default_233 = None
        add_tensor_260 = torch.ops.aten.add.Tensor(
            mul_tensor_412, unsqueeze_default_235
        )
        mul_tensor_412 = unsqueeze_default_235 = None
        relu_default_58 = torch.ops.aten.relu.default(add_tensor_260)
        add_tensor_260 = None
        convolution_default_110 = torch.ops.aten.convolution.default(
            relu_default_58,
            primals_235,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_111 = torch.ops.aten.convolution.default(
            convolution_default_110,
            primals_236,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_118 = torch.ops.aten.var.correction(
            convolution_default_111, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_59 = torch.ops.aten.mean.dim(convolution_default_111, [0, 2, 3], True)
        add_tensor_261 = torch.ops.aten.add.Tensor(var_correction_118, 0.001)
        var_correction_118 = None
        sqrt_default_59 = torch.ops.aten.sqrt.default(add_tensor_261)
        add_tensor_261 = None
        reciprocal_default_59 = torch.ops.aten.reciprocal.default(sqrt_default_59)
        sqrt_default_59 = None
        sub_tensor_59 = torch.ops.aten.sub.Tensor(convolution_default_111, mean_dim_59)
        mul_tensor_413 = torch.ops.aten.mul.Tensor(sub_tensor_59, reciprocal_default_59)
        sub_tensor_59 = None
        squeeze_dim_354 = torch.ops.aten.squeeze.dim(mean_dim_59, 3)
        mean_dim_59 = None
        squeeze_dim_355 = torch.ops.aten.squeeze.dim(squeeze_dim_354, 2)
        squeeze_dim_354 = None
        squeeze_dim_356 = torch.ops.aten.squeeze.dim(squeeze_dim_355, 0)
        squeeze_dim_355 = None
        squeeze_dim_357 = torch.ops.aten.squeeze.dim(reciprocal_default_59, 3)
        reciprocal_default_59 = None
        squeeze_dim_358 = torch.ops.aten.squeeze.dim(squeeze_dim_357, 2)
        squeeze_dim_357 = None
        squeeze_dim_359 = torch.ops.aten.squeeze.dim(squeeze_dim_358, 0)
        squeeze_dim_358 = None
        unsqueeze_default_236 = torch.ops.aten.unsqueeze.default(primals_237, -1)
        unsqueeze_default_237 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_236, -1
        )
        unsqueeze_default_236 = None
        unsqueeze_default_238 = torch.ops.aten.unsqueeze.default(primals_238, -1)
        primals_238 = None
        unsqueeze_default_239 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_238, -1
        )
        unsqueeze_default_238 = None
        mul_tensor_419 = torch.ops.aten.mul.Tensor(
            mul_tensor_413, unsqueeze_default_237
        )
        mul_tensor_413 = unsqueeze_default_237 = None
        add_tensor_264 = torch.ops.aten.add.Tensor(
            mul_tensor_419, unsqueeze_default_239
        )
        mul_tensor_419 = unsqueeze_default_239 = None
        add_tensor_265 = torch.ops.aten.add.Tensor(add_tensor_264, add_tensor_220)
        add_tensor_264 = add_tensor_220 = None
        cat_default_6 = torch.ops.aten.cat.default(
            [
                add_tensor_216,
                add_tensor_237,
                add_tensor_254,
                add_tensor_255,
                add_tensor_256,
                add_tensor_265,
            ],
            1,
        )
        add_tensor_216 = (
            add_tensor_237
        ) = add_tensor_254 = add_tensor_255 = add_tensor_256 = add_tensor_265 = None
        convolution_default_112 = torch.ops.aten.convolution.default(
            relu_default_48, primals_239, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_120 = torch.ops.aten.var.correction(
            convolution_default_112, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_60 = torch.ops.aten.mean.dim(convolution_default_112, [0, 2, 3], True)
        add_tensor_266 = torch.ops.aten.add.Tensor(var_correction_120, 0.001)
        var_correction_120 = None
        sqrt_default_60 = torch.ops.aten.sqrt.default(add_tensor_266)
        add_tensor_266 = None
        reciprocal_default_60 = torch.ops.aten.reciprocal.default(sqrt_default_60)
        sqrt_default_60 = None
        sub_tensor_60 = torch.ops.aten.sub.Tensor(convolution_default_112, mean_dim_60)
        mul_tensor_420 = torch.ops.aten.mul.Tensor(sub_tensor_60, reciprocal_default_60)
        sub_tensor_60 = None
        unsqueeze_default_240 = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_default_241 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_240, -1
        )
        unsqueeze_default_240 = None
        unsqueeze_default_242 = torch.ops.aten.unsqueeze.default(primals_241, -1)
        unsqueeze_default_243 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_242, -1
        )
        unsqueeze_default_242 = None
        mul_tensor_426 = torch.ops.aten.mul.Tensor(
            mul_tensor_420, unsqueeze_default_241
        )
        mul_tensor_420 = unsqueeze_default_241 = None
        add_tensor_269 = torch.ops.aten.add.Tensor(
            mul_tensor_426, unsqueeze_default_243
        )
        mul_tensor_426 = unsqueeze_default_243 = None
        relu_default_60 = torch.ops.aten.relu.default(cat_default_6)
        cat_default_6 = None
        convolution_default_113 = torch.ops.aten.convolution.default(
            relu_default_60, primals_242, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_122 = torch.ops.aten.var.correction(
            convolution_default_113, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_61 = torch.ops.aten.mean.dim(convolution_default_113, [0, 2, 3], True)
        add_tensor_270 = torch.ops.aten.add.Tensor(var_correction_122, 0.001)
        var_correction_122 = None
        sqrt_default_61 = torch.ops.aten.sqrt.default(add_tensor_270)
        add_tensor_270 = None
        reciprocal_default_61 = torch.ops.aten.reciprocal.default(sqrt_default_61)
        sqrt_default_61 = None
        sub_tensor_61 = torch.ops.aten.sub.Tensor(convolution_default_113, mean_dim_61)
        mul_tensor_427 = torch.ops.aten.mul.Tensor(sub_tensor_61, reciprocal_default_61)
        sub_tensor_61 = None
        unsqueeze_default_244 = torch.ops.aten.unsqueeze.default(primals_243, -1)
        unsqueeze_default_245 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_244, -1
        )
        unsqueeze_default_244 = None
        unsqueeze_default_246 = torch.ops.aten.unsqueeze.default(primals_244, -1)
        unsqueeze_default_247 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_246, -1
        )
        unsqueeze_default_246 = None
        mul_tensor_433 = torch.ops.aten.mul.Tensor(
            mul_tensor_427, unsqueeze_default_245
        )
        mul_tensor_427 = unsqueeze_default_245 = None
        add_tensor_273 = torch.ops.aten.add.Tensor(
            mul_tensor_433, unsqueeze_default_247
        )
        mul_tensor_433 = unsqueeze_default_247 = None
        relu_default_61 = torch.ops.aten.relu.default(add_tensor_273)
        convolution_default_114 = torch.ops.aten.convolution.default(
            relu_default_61,
            primals_245,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_115 = torch.ops.aten.convolution.default(
            convolution_default_114,
            primals_246,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_124 = torch.ops.aten.var.correction(
            convolution_default_115, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_62 = torch.ops.aten.mean.dim(convolution_default_115, [0, 2, 3], True)
        add_tensor_274 = torch.ops.aten.add.Tensor(var_correction_124, 0.001)
        var_correction_124 = None
        sqrt_default_62 = torch.ops.aten.sqrt.default(add_tensor_274)
        add_tensor_274 = None
        reciprocal_default_62 = torch.ops.aten.reciprocal.default(sqrt_default_62)
        sqrt_default_62 = None
        sub_tensor_62 = torch.ops.aten.sub.Tensor(convolution_default_115, mean_dim_62)
        mul_tensor_434 = torch.ops.aten.mul.Tensor(sub_tensor_62, reciprocal_default_62)
        sub_tensor_62 = None
        squeeze_dim_372 = torch.ops.aten.squeeze.dim(mean_dim_62, 3)
        mean_dim_62 = None
        squeeze_dim_373 = torch.ops.aten.squeeze.dim(squeeze_dim_372, 2)
        squeeze_dim_372 = None
        squeeze_dim_374 = torch.ops.aten.squeeze.dim(squeeze_dim_373, 0)
        squeeze_dim_373 = None
        squeeze_dim_375 = torch.ops.aten.squeeze.dim(reciprocal_default_62, 3)
        reciprocal_default_62 = None
        squeeze_dim_376 = torch.ops.aten.squeeze.dim(squeeze_dim_375, 2)
        squeeze_dim_375 = None
        squeeze_dim_377 = torch.ops.aten.squeeze.dim(squeeze_dim_376, 0)
        squeeze_dim_376 = None
        unsqueeze_default_248 = torch.ops.aten.unsqueeze.default(primals_247, -1)
        unsqueeze_default_249 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_248, -1
        )
        unsqueeze_default_248 = None
        unsqueeze_default_250 = torch.ops.aten.unsqueeze.default(primals_248, -1)
        primals_248 = None
        unsqueeze_default_251 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_250, -1
        )
        unsqueeze_default_250 = None
        mul_tensor_440 = torch.ops.aten.mul.Tensor(
            mul_tensor_434, unsqueeze_default_249
        )
        mul_tensor_434 = unsqueeze_default_249 = None
        add_tensor_277 = torch.ops.aten.add.Tensor(
            mul_tensor_440, unsqueeze_default_251
        )
        mul_tensor_440 = unsqueeze_default_251 = None
        relu_default_62 = torch.ops.aten.relu.default(add_tensor_277)
        add_tensor_277 = None
        convolution_default_116 = torch.ops.aten.convolution.default(
            relu_default_62,
            primals_249,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_117 = torch.ops.aten.convolution.default(
            convolution_default_116,
            primals_250,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_126 = torch.ops.aten.var.correction(
            convolution_default_117, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_63 = torch.ops.aten.mean.dim(convolution_default_117, [0, 2, 3], True)
        add_tensor_278 = torch.ops.aten.add.Tensor(var_correction_126, 0.001)
        var_correction_126 = None
        sqrt_default_63 = torch.ops.aten.sqrt.default(add_tensor_278)
        add_tensor_278 = None
        reciprocal_default_63 = torch.ops.aten.reciprocal.default(sqrt_default_63)
        sqrt_default_63 = None
        sub_tensor_63 = torch.ops.aten.sub.Tensor(convolution_default_117, mean_dim_63)
        mul_tensor_441 = torch.ops.aten.mul.Tensor(sub_tensor_63, reciprocal_default_63)
        sub_tensor_63 = None
        squeeze_dim_378 = torch.ops.aten.squeeze.dim(mean_dim_63, 3)
        mean_dim_63 = None
        squeeze_dim_379 = torch.ops.aten.squeeze.dim(squeeze_dim_378, 2)
        squeeze_dim_378 = None
        squeeze_dim_380 = torch.ops.aten.squeeze.dim(squeeze_dim_379, 0)
        squeeze_dim_379 = None
        squeeze_dim_381 = torch.ops.aten.squeeze.dim(reciprocal_default_63, 3)
        reciprocal_default_63 = None
        squeeze_dim_382 = torch.ops.aten.squeeze.dim(squeeze_dim_381, 2)
        squeeze_dim_381 = None
        squeeze_dim_383 = torch.ops.aten.squeeze.dim(squeeze_dim_382, 0)
        squeeze_dim_382 = None
        unsqueeze_default_252 = torch.ops.aten.unsqueeze.default(primals_251, -1)
        unsqueeze_default_253 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_252, -1
        )
        unsqueeze_default_252 = None
        unsqueeze_default_254 = torch.ops.aten.unsqueeze.default(primals_252, -1)
        primals_252 = None
        unsqueeze_default_255 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_254, -1
        )
        unsqueeze_default_254 = None
        mul_tensor_447 = torch.ops.aten.mul.Tensor(
            mul_tensor_441, unsqueeze_default_253
        )
        mul_tensor_441 = unsqueeze_default_253 = None
        add_tensor_281 = torch.ops.aten.add.Tensor(
            mul_tensor_447, unsqueeze_default_255
        )
        mul_tensor_447 = unsqueeze_default_255 = None
        relu_default_63 = torch.ops.aten.relu.default(add_tensor_269)
        convolution_default_118 = torch.ops.aten.convolution.default(
            relu_default_63,
            primals_253,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_119 = torch.ops.aten.convolution.default(
            convolution_default_118,
            primals_254,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_128 = torch.ops.aten.var.correction(
            convolution_default_119, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_64 = torch.ops.aten.mean.dim(convolution_default_119, [0, 2, 3], True)
        add_tensor_282 = torch.ops.aten.add.Tensor(var_correction_128, 0.001)
        var_correction_128 = None
        sqrt_default_64 = torch.ops.aten.sqrt.default(add_tensor_282)
        add_tensor_282 = None
        reciprocal_default_64 = torch.ops.aten.reciprocal.default(sqrt_default_64)
        sqrt_default_64 = None
        sub_tensor_64 = torch.ops.aten.sub.Tensor(convolution_default_119, mean_dim_64)
        mul_tensor_448 = torch.ops.aten.mul.Tensor(sub_tensor_64, reciprocal_default_64)
        sub_tensor_64 = None
        squeeze_dim_384 = torch.ops.aten.squeeze.dim(mean_dim_64, 3)
        mean_dim_64 = None
        squeeze_dim_385 = torch.ops.aten.squeeze.dim(squeeze_dim_384, 2)
        squeeze_dim_384 = None
        squeeze_dim_386 = torch.ops.aten.squeeze.dim(squeeze_dim_385, 0)
        squeeze_dim_385 = None
        squeeze_dim_387 = torch.ops.aten.squeeze.dim(reciprocal_default_64, 3)
        reciprocal_default_64 = None
        squeeze_dim_388 = torch.ops.aten.squeeze.dim(squeeze_dim_387, 2)
        squeeze_dim_387 = None
        squeeze_dim_389 = torch.ops.aten.squeeze.dim(squeeze_dim_388, 0)
        squeeze_dim_388 = None
        unsqueeze_default_256 = torch.ops.aten.unsqueeze.default(primals_255, -1)
        unsqueeze_default_257 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_256, -1
        )
        unsqueeze_default_256 = None
        unsqueeze_default_258 = torch.ops.aten.unsqueeze.default(primals_256, -1)
        primals_256 = None
        unsqueeze_default_259 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_258, -1
        )
        unsqueeze_default_258 = None
        mul_tensor_454 = torch.ops.aten.mul.Tensor(
            mul_tensor_448, unsqueeze_default_257
        )
        mul_tensor_448 = unsqueeze_default_257 = None
        add_tensor_285 = torch.ops.aten.add.Tensor(
            mul_tensor_454, unsqueeze_default_259
        )
        mul_tensor_454 = unsqueeze_default_259 = None
        relu_default_64 = torch.ops.aten.relu.default(add_tensor_285)
        add_tensor_285 = None
        convolution_default_120 = torch.ops.aten.convolution.default(
            relu_default_64,
            primals_257,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_121 = torch.ops.aten.convolution.default(
            convolution_default_120,
            primals_258,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_130 = torch.ops.aten.var.correction(
            convolution_default_121, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_65 = torch.ops.aten.mean.dim(convolution_default_121, [0, 2, 3], True)
        add_tensor_286 = torch.ops.aten.add.Tensor(var_correction_130, 0.001)
        var_correction_130 = None
        sqrt_default_65 = torch.ops.aten.sqrt.default(add_tensor_286)
        add_tensor_286 = None
        reciprocal_default_65 = torch.ops.aten.reciprocal.default(sqrt_default_65)
        sqrt_default_65 = None
        sub_tensor_65 = torch.ops.aten.sub.Tensor(convolution_default_121, mean_dim_65)
        mul_tensor_455 = torch.ops.aten.mul.Tensor(sub_tensor_65, reciprocal_default_65)
        sub_tensor_65 = None
        squeeze_dim_390 = torch.ops.aten.squeeze.dim(mean_dim_65, 3)
        mean_dim_65 = None
        squeeze_dim_391 = torch.ops.aten.squeeze.dim(squeeze_dim_390, 2)
        squeeze_dim_390 = None
        squeeze_dim_392 = torch.ops.aten.squeeze.dim(squeeze_dim_391, 0)
        squeeze_dim_391 = None
        squeeze_dim_393 = torch.ops.aten.squeeze.dim(reciprocal_default_65, 3)
        reciprocal_default_65 = None
        squeeze_dim_394 = torch.ops.aten.squeeze.dim(squeeze_dim_393, 2)
        squeeze_dim_393 = None
        squeeze_dim_395 = torch.ops.aten.squeeze.dim(squeeze_dim_394, 0)
        squeeze_dim_394 = None
        unsqueeze_default_260 = torch.ops.aten.unsqueeze.default(primals_259, -1)
        unsqueeze_default_261 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_260, -1
        )
        unsqueeze_default_260 = None
        unsqueeze_default_262 = torch.ops.aten.unsqueeze.default(primals_260, -1)
        primals_260 = None
        unsqueeze_default_263 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_262, -1
        )
        unsqueeze_default_262 = None
        mul_tensor_461 = torch.ops.aten.mul.Tensor(
            mul_tensor_455, unsqueeze_default_261
        )
        mul_tensor_455 = unsqueeze_default_261 = None
        add_tensor_289 = torch.ops.aten.add.Tensor(
            mul_tensor_461, unsqueeze_default_263
        )
        mul_tensor_461 = unsqueeze_default_263 = None
        add_tensor_290 = torch.ops.aten.add.Tensor(add_tensor_281, add_tensor_289)
        add_tensor_281 = add_tensor_289 = None
        convolution_default_122 = torch.ops.aten.convolution.default(
            relu_default_63,
            primals_261,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_123 = torch.ops.aten.convolution.default(
            convolution_default_122,
            primals_262,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_132 = torch.ops.aten.var.correction(
            convolution_default_123, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_66 = torch.ops.aten.mean.dim(convolution_default_123, [0, 2, 3], True)
        add_tensor_291 = torch.ops.aten.add.Tensor(var_correction_132, 0.001)
        var_correction_132 = None
        sqrt_default_66 = torch.ops.aten.sqrt.default(add_tensor_291)
        add_tensor_291 = None
        reciprocal_default_66 = torch.ops.aten.reciprocal.default(sqrt_default_66)
        sqrt_default_66 = None
        sub_tensor_66 = torch.ops.aten.sub.Tensor(convolution_default_123, mean_dim_66)
        mul_tensor_462 = torch.ops.aten.mul.Tensor(sub_tensor_66, reciprocal_default_66)
        sub_tensor_66 = None
        squeeze_dim_396 = torch.ops.aten.squeeze.dim(mean_dim_66, 3)
        mean_dim_66 = None
        squeeze_dim_397 = torch.ops.aten.squeeze.dim(squeeze_dim_396, 2)
        squeeze_dim_396 = None
        squeeze_dim_398 = torch.ops.aten.squeeze.dim(squeeze_dim_397, 0)
        squeeze_dim_397 = None
        squeeze_dim_399 = torch.ops.aten.squeeze.dim(reciprocal_default_66, 3)
        reciprocal_default_66 = None
        squeeze_dim_400 = torch.ops.aten.squeeze.dim(squeeze_dim_399, 2)
        squeeze_dim_399 = None
        squeeze_dim_401 = torch.ops.aten.squeeze.dim(squeeze_dim_400, 0)
        squeeze_dim_400 = None
        unsqueeze_default_264 = torch.ops.aten.unsqueeze.default(primals_263, -1)
        unsqueeze_default_265 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_264, -1
        )
        unsqueeze_default_264 = None
        unsqueeze_default_266 = torch.ops.aten.unsqueeze.default(primals_264, -1)
        primals_264 = None
        unsqueeze_default_267 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_266, -1
        )
        unsqueeze_default_266 = None
        mul_tensor_468 = torch.ops.aten.mul.Tensor(
            mul_tensor_462, unsqueeze_default_265
        )
        mul_tensor_462 = unsqueeze_default_265 = None
        add_tensor_294 = torch.ops.aten.add.Tensor(
            mul_tensor_468, unsqueeze_default_267
        )
        mul_tensor_468 = unsqueeze_default_267 = None
        relu_default_66 = torch.ops.aten.relu.default(add_tensor_294)
        add_tensor_294 = None
        convolution_default_124 = torch.ops.aten.convolution.default(
            relu_default_66,
            primals_265,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_125 = torch.ops.aten.convolution.default(
            convolution_default_124,
            primals_266,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_134 = torch.ops.aten.var.correction(
            convolution_default_125, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_67 = torch.ops.aten.mean.dim(convolution_default_125, [0, 2, 3], True)
        add_tensor_295 = torch.ops.aten.add.Tensor(var_correction_134, 0.001)
        var_correction_134 = None
        sqrt_default_67 = torch.ops.aten.sqrt.default(add_tensor_295)
        add_tensor_295 = None
        reciprocal_default_67 = torch.ops.aten.reciprocal.default(sqrt_default_67)
        sqrt_default_67 = None
        sub_tensor_67 = torch.ops.aten.sub.Tensor(convolution_default_125, mean_dim_67)
        mul_tensor_469 = torch.ops.aten.mul.Tensor(sub_tensor_67, reciprocal_default_67)
        sub_tensor_67 = None
        squeeze_dim_402 = torch.ops.aten.squeeze.dim(mean_dim_67, 3)
        mean_dim_67 = None
        squeeze_dim_403 = torch.ops.aten.squeeze.dim(squeeze_dim_402, 2)
        squeeze_dim_402 = None
        squeeze_dim_404 = torch.ops.aten.squeeze.dim(squeeze_dim_403, 0)
        squeeze_dim_403 = None
        squeeze_dim_405 = torch.ops.aten.squeeze.dim(reciprocal_default_67, 3)
        reciprocal_default_67 = None
        squeeze_dim_406 = torch.ops.aten.squeeze.dim(squeeze_dim_405, 2)
        squeeze_dim_405 = None
        squeeze_dim_407 = torch.ops.aten.squeeze.dim(squeeze_dim_406, 0)
        squeeze_dim_406 = None
        unsqueeze_default_268 = torch.ops.aten.unsqueeze.default(primals_267, -1)
        unsqueeze_default_269 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_268, -1
        )
        unsqueeze_default_268 = None
        unsqueeze_default_270 = torch.ops.aten.unsqueeze.default(primals_268, -1)
        primals_268 = None
        unsqueeze_default_271 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_270, -1
        )
        unsqueeze_default_270 = None
        mul_tensor_475 = torch.ops.aten.mul.Tensor(
            mul_tensor_469, unsqueeze_default_269
        )
        mul_tensor_469 = unsqueeze_default_269 = None
        add_tensor_298 = torch.ops.aten.add.Tensor(
            mul_tensor_475, unsqueeze_default_271
        )
        mul_tensor_475 = unsqueeze_default_271 = None
        convolution_default_126 = torch.ops.aten.convolution.default(
            relu_default_63,
            primals_269,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_63 = None
        convolution_default_127 = torch.ops.aten.convolution.default(
            convolution_default_126,
            primals_270,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_136 = torch.ops.aten.var.correction(
            convolution_default_127, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_68 = torch.ops.aten.mean.dim(convolution_default_127, [0, 2, 3], True)
        add_tensor_299 = torch.ops.aten.add.Tensor(var_correction_136, 0.001)
        var_correction_136 = None
        sqrt_default_68 = torch.ops.aten.sqrt.default(add_tensor_299)
        add_tensor_299 = None
        reciprocal_default_68 = torch.ops.aten.reciprocal.default(sqrt_default_68)
        sqrt_default_68 = None
        sub_tensor_68 = torch.ops.aten.sub.Tensor(convolution_default_127, mean_dim_68)
        mul_tensor_476 = torch.ops.aten.mul.Tensor(sub_tensor_68, reciprocal_default_68)
        sub_tensor_68 = None
        squeeze_dim_408 = torch.ops.aten.squeeze.dim(mean_dim_68, 3)
        mean_dim_68 = None
        squeeze_dim_409 = torch.ops.aten.squeeze.dim(squeeze_dim_408, 2)
        squeeze_dim_408 = None
        squeeze_dim_410 = torch.ops.aten.squeeze.dim(squeeze_dim_409, 0)
        squeeze_dim_409 = None
        squeeze_dim_411 = torch.ops.aten.squeeze.dim(reciprocal_default_68, 3)
        reciprocal_default_68 = None
        squeeze_dim_412 = torch.ops.aten.squeeze.dim(squeeze_dim_411, 2)
        squeeze_dim_411 = None
        squeeze_dim_413 = torch.ops.aten.squeeze.dim(squeeze_dim_412, 0)
        squeeze_dim_412 = None
        unsqueeze_default_272 = torch.ops.aten.unsqueeze.default(primals_271, -1)
        unsqueeze_default_273 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_272, -1
        )
        unsqueeze_default_272 = None
        unsqueeze_default_274 = torch.ops.aten.unsqueeze.default(primals_272, -1)
        primals_272 = None
        unsqueeze_default_275 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_274, -1
        )
        unsqueeze_default_274 = None
        mul_tensor_482 = torch.ops.aten.mul.Tensor(
            mul_tensor_476, unsqueeze_default_273
        )
        mul_tensor_476 = unsqueeze_default_273 = None
        add_tensor_302 = torch.ops.aten.add.Tensor(
            mul_tensor_482, unsqueeze_default_275
        )
        mul_tensor_482 = unsqueeze_default_275 = None
        relu_default_68 = torch.ops.aten.relu.default(add_tensor_302)
        add_tensor_302 = None
        convolution_default_128 = torch.ops.aten.convolution.default(
            relu_default_68,
            primals_273,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_129 = torch.ops.aten.convolution.default(
            convolution_default_128,
            primals_274,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_138 = torch.ops.aten.var.correction(
            convolution_default_129, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_69 = torch.ops.aten.mean.dim(convolution_default_129, [0, 2, 3], True)
        add_tensor_303 = torch.ops.aten.add.Tensor(var_correction_138, 0.001)
        var_correction_138 = None
        sqrt_default_69 = torch.ops.aten.sqrt.default(add_tensor_303)
        add_tensor_303 = None
        reciprocal_default_69 = torch.ops.aten.reciprocal.default(sqrt_default_69)
        sqrt_default_69 = None
        sub_tensor_69 = torch.ops.aten.sub.Tensor(convolution_default_129, mean_dim_69)
        mul_tensor_483 = torch.ops.aten.mul.Tensor(sub_tensor_69, reciprocal_default_69)
        sub_tensor_69 = None
        squeeze_dim_414 = torch.ops.aten.squeeze.dim(mean_dim_69, 3)
        mean_dim_69 = None
        squeeze_dim_415 = torch.ops.aten.squeeze.dim(squeeze_dim_414, 2)
        squeeze_dim_414 = None
        squeeze_dim_416 = torch.ops.aten.squeeze.dim(squeeze_dim_415, 0)
        squeeze_dim_415 = None
        squeeze_dim_417 = torch.ops.aten.squeeze.dim(reciprocal_default_69, 3)
        reciprocal_default_69 = None
        squeeze_dim_418 = torch.ops.aten.squeeze.dim(squeeze_dim_417, 2)
        squeeze_dim_417 = None
        squeeze_dim_419 = torch.ops.aten.squeeze.dim(squeeze_dim_418, 0)
        squeeze_dim_418 = None
        unsqueeze_default_276 = torch.ops.aten.unsqueeze.default(primals_275, -1)
        unsqueeze_default_277 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_276, -1
        )
        unsqueeze_default_276 = None
        unsqueeze_default_278 = torch.ops.aten.unsqueeze.default(primals_276, -1)
        primals_276 = None
        unsqueeze_default_279 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_278, -1
        )
        unsqueeze_default_278 = None
        mul_tensor_489 = torch.ops.aten.mul.Tensor(
            mul_tensor_483, unsqueeze_default_277
        )
        mul_tensor_483 = unsqueeze_default_277 = None
        add_tensor_306 = torch.ops.aten.add.Tensor(
            mul_tensor_489, unsqueeze_default_279
        )
        mul_tensor_489 = unsqueeze_default_279 = None
        add_tensor_307 = torch.ops.aten.add.Tensor(add_tensor_298, add_tensor_306)
        add_tensor_298 = add_tensor_306 = None
        avg_pool2d_default_17 = torch.ops.aten.avg_pool2d.default(
            add_tensor_273, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_308 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_17, add_tensor_269
        )
        avg_pool2d_default_17 = None
        avg_pool2d_default_18 = torch.ops.aten.avg_pool2d.default(
            add_tensor_269, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_309 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_18, avg_pool2d_default_18
        )
        avg_pool2d_default_18 = None
        convolution_default_130 = torch.ops.aten.convolution.default(
            relu_default_61,
            primals_277,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_61 = None
        convolution_default_131 = torch.ops.aten.convolution.default(
            convolution_default_130,
            primals_278,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_140 = torch.ops.aten.var.correction(
            convolution_default_131, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_70 = torch.ops.aten.mean.dim(convolution_default_131, [0, 2, 3], True)
        add_tensor_310 = torch.ops.aten.add.Tensor(var_correction_140, 0.001)
        var_correction_140 = None
        sqrt_default_70 = torch.ops.aten.sqrt.default(add_tensor_310)
        add_tensor_310 = None
        reciprocal_default_70 = torch.ops.aten.reciprocal.default(sqrt_default_70)
        sqrt_default_70 = None
        sub_tensor_70 = torch.ops.aten.sub.Tensor(convolution_default_131, mean_dim_70)
        mul_tensor_490 = torch.ops.aten.mul.Tensor(sub_tensor_70, reciprocal_default_70)
        sub_tensor_70 = None
        squeeze_dim_420 = torch.ops.aten.squeeze.dim(mean_dim_70, 3)
        mean_dim_70 = None
        squeeze_dim_421 = torch.ops.aten.squeeze.dim(squeeze_dim_420, 2)
        squeeze_dim_420 = None
        squeeze_dim_422 = torch.ops.aten.squeeze.dim(squeeze_dim_421, 0)
        squeeze_dim_421 = None
        squeeze_dim_423 = torch.ops.aten.squeeze.dim(reciprocal_default_70, 3)
        reciprocal_default_70 = None
        squeeze_dim_424 = torch.ops.aten.squeeze.dim(squeeze_dim_423, 2)
        squeeze_dim_423 = None
        squeeze_dim_425 = torch.ops.aten.squeeze.dim(squeeze_dim_424, 0)
        squeeze_dim_424 = None
        unsqueeze_default_280 = torch.ops.aten.unsqueeze.default(primals_279, -1)
        unsqueeze_default_281 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_280, -1
        )
        unsqueeze_default_280 = None
        unsqueeze_default_282 = torch.ops.aten.unsqueeze.default(primals_280, -1)
        primals_280 = None
        unsqueeze_default_283 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_282, -1
        )
        unsqueeze_default_282 = None
        mul_tensor_496 = torch.ops.aten.mul.Tensor(
            mul_tensor_490, unsqueeze_default_281
        )
        mul_tensor_490 = unsqueeze_default_281 = None
        add_tensor_313 = torch.ops.aten.add.Tensor(
            mul_tensor_496, unsqueeze_default_283
        )
        mul_tensor_496 = unsqueeze_default_283 = None
        relu_default_70 = torch.ops.aten.relu.default(add_tensor_313)
        add_tensor_313 = None
        convolution_default_132 = torch.ops.aten.convolution.default(
            relu_default_70,
            primals_281,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_133 = torch.ops.aten.convolution.default(
            convolution_default_132,
            primals_282,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_142 = torch.ops.aten.var.correction(
            convolution_default_133, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_71 = torch.ops.aten.mean.dim(convolution_default_133, [0, 2, 3], True)
        add_tensor_314 = torch.ops.aten.add.Tensor(var_correction_142, 0.001)
        var_correction_142 = None
        sqrt_default_71 = torch.ops.aten.sqrt.default(add_tensor_314)
        add_tensor_314 = None
        reciprocal_default_71 = torch.ops.aten.reciprocal.default(sqrt_default_71)
        sqrt_default_71 = None
        sub_tensor_71 = torch.ops.aten.sub.Tensor(convolution_default_133, mean_dim_71)
        mul_tensor_497 = torch.ops.aten.mul.Tensor(sub_tensor_71, reciprocal_default_71)
        sub_tensor_71 = None
        squeeze_dim_426 = torch.ops.aten.squeeze.dim(mean_dim_71, 3)
        mean_dim_71 = None
        squeeze_dim_427 = torch.ops.aten.squeeze.dim(squeeze_dim_426, 2)
        squeeze_dim_426 = None
        squeeze_dim_428 = torch.ops.aten.squeeze.dim(squeeze_dim_427, 0)
        squeeze_dim_427 = None
        squeeze_dim_429 = torch.ops.aten.squeeze.dim(reciprocal_default_71, 3)
        reciprocal_default_71 = None
        squeeze_dim_430 = torch.ops.aten.squeeze.dim(squeeze_dim_429, 2)
        squeeze_dim_429 = None
        squeeze_dim_431 = torch.ops.aten.squeeze.dim(squeeze_dim_430, 0)
        squeeze_dim_430 = None
        unsqueeze_default_284 = torch.ops.aten.unsqueeze.default(primals_283, -1)
        unsqueeze_default_285 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_284, -1
        )
        unsqueeze_default_284 = None
        unsqueeze_default_286 = torch.ops.aten.unsqueeze.default(primals_284, -1)
        primals_284 = None
        unsqueeze_default_287 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_286, -1
        )
        unsqueeze_default_286 = None
        mul_tensor_503 = torch.ops.aten.mul.Tensor(
            mul_tensor_497, unsqueeze_default_285
        )
        mul_tensor_497 = unsqueeze_default_285 = None
        add_tensor_317 = torch.ops.aten.add.Tensor(
            mul_tensor_503, unsqueeze_default_287
        )
        mul_tensor_503 = unsqueeze_default_287 = None
        add_tensor_318 = torch.ops.aten.add.Tensor(add_tensor_317, add_tensor_273)
        add_tensor_317 = add_tensor_273 = None
        cat_default_7 = torch.ops.aten.cat.default(
            [
                add_tensor_269,
                add_tensor_290,
                add_tensor_307,
                add_tensor_308,
                add_tensor_309,
                add_tensor_318,
            ],
            1,
        )
        add_tensor_269 = (
            add_tensor_290
        ) = add_tensor_307 = add_tensor_308 = add_tensor_309 = add_tensor_318 = None
        convolution_default_134 = torch.ops.aten.convolution.default(
            relu_default_60, primals_285, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_144 = torch.ops.aten.var.correction(
            convolution_default_134, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_72 = torch.ops.aten.mean.dim(convolution_default_134, [0, 2, 3], True)
        add_tensor_319 = torch.ops.aten.add.Tensor(var_correction_144, 0.001)
        var_correction_144 = None
        sqrt_default_72 = torch.ops.aten.sqrt.default(add_tensor_319)
        add_tensor_319 = None
        reciprocal_default_72 = torch.ops.aten.reciprocal.default(sqrt_default_72)
        sqrt_default_72 = None
        sub_tensor_72 = torch.ops.aten.sub.Tensor(convolution_default_134, mean_dim_72)
        mul_tensor_504 = torch.ops.aten.mul.Tensor(sub_tensor_72, reciprocal_default_72)
        sub_tensor_72 = None
        unsqueeze_default_288 = torch.ops.aten.unsqueeze.default(primals_286, -1)
        unsqueeze_default_289 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_288, -1
        )
        unsqueeze_default_288 = None
        unsqueeze_default_290 = torch.ops.aten.unsqueeze.default(primals_287, -1)
        unsqueeze_default_291 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_290, -1
        )
        unsqueeze_default_290 = None
        mul_tensor_510 = torch.ops.aten.mul.Tensor(
            mul_tensor_504, unsqueeze_default_289
        )
        mul_tensor_504 = unsqueeze_default_289 = None
        add_tensor_322 = torch.ops.aten.add.Tensor(
            mul_tensor_510, unsqueeze_default_291
        )
        mul_tensor_510 = unsqueeze_default_291 = None
        relu_default_72 = torch.ops.aten.relu.default(cat_default_7)
        cat_default_7 = None
        convolution_default_135 = torch.ops.aten.convolution.default(
            relu_default_72, primals_288, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_146 = torch.ops.aten.var.correction(
            convolution_default_135, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_73 = torch.ops.aten.mean.dim(convolution_default_135, [0, 2, 3], True)
        add_tensor_323 = torch.ops.aten.add.Tensor(var_correction_146, 0.001)
        var_correction_146 = None
        sqrt_default_73 = torch.ops.aten.sqrt.default(add_tensor_323)
        add_tensor_323 = None
        reciprocal_default_73 = torch.ops.aten.reciprocal.default(sqrt_default_73)
        sqrt_default_73 = None
        sub_tensor_73 = torch.ops.aten.sub.Tensor(convolution_default_135, mean_dim_73)
        mul_tensor_511 = torch.ops.aten.mul.Tensor(sub_tensor_73, reciprocal_default_73)
        sub_tensor_73 = None
        unsqueeze_default_292 = torch.ops.aten.unsqueeze.default(primals_289, -1)
        unsqueeze_default_293 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_292, -1
        )
        unsqueeze_default_292 = None
        unsqueeze_default_294 = torch.ops.aten.unsqueeze.default(primals_290, -1)
        unsqueeze_default_295 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_294, -1
        )
        unsqueeze_default_294 = None
        mul_tensor_517 = torch.ops.aten.mul.Tensor(
            mul_tensor_511, unsqueeze_default_293
        )
        mul_tensor_511 = unsqueeze_default_293 = None
        add_tensor_326 = torch.ops.aten.add.Tensor(
            mul_tensor_517, unsqueeze_default_295
        )
        mul_tensor_517 = unsqueeze_default_295 = None
        relu_default_73 = torch.ops.aten.relu.default(add_tensor_326)
        convolution_default_136 = torch.ops.aten.convolution.default(
            relu_default_73,
            primals_291,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_137 = torch.ops.aten.convolution.default(
            convolution_default_136,
            primals_292,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_148 = torch.ops.aten.var.correction(
            convolution_default_137, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_74 = torch.ops.aten.mean.dim(convolution_default_137, [0, 2, 3], True)
        add_tensor_327 = torch.ops.aten.add.Tensor(var_correction_148, 0.001)
        var_correction_148 = None
        sqrt_default_74 = torch.ops.aten.sqrt.default(add_tensor_327)
        add_tensor_327 = None
        reciprocal_default_74 = torch.ops.aten.reciprocal.default(sqrt_default_74)
        sqrt_default_74 = None
        sub_tensor_74 = torch.ops.aten.sub.Tensor(convolution_default_137, mean_dim_74)
        mul_tensor_518 = torch.ops.aten.mul.Tensor(sub_tensor_74, reciprocal_default_74)
        sub_tensor_74 = None
        squeeze_dim_444 = torch.ops.aten.squeeze.dim(mean_dim_74, 3)
        mean_dim_74 = None
        squeeze_dim_445 = torch.ops.aten.squeeze.dim(squeeze_dim_444, 2)
        squeeze_dim_444 = None
        squeeze_dim_446 = torch.ops.aten.squeeze.dim(squeeze_dim_445, 0)
        squeeze_dim_445 = None
        squeeze_dim_447 = torch.ops.aten.squeeze.dim(reciprocal_default_74, 3)
        reciprocal_default_74 = None
        squeeze_dim_448 = torch.ops.aten.squeeze.dim(squeeze_dim_447, 2)
        squeeze_dim_447 = None
        squeeze_dim_449 = torch.ops.aten.squeeze.dim(squeeze_dim_448, 0)
        squeeze_dim_448 = None
        unsqueeze_default_296 = torch.ops.aten.unsqueeze.default(primals_293, -1)
        unsqueeze_default_297 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_296, -1
        )
        unsqueeze_default_296 = None
        unsqueeze_default_298 = torch.ops.aten.unsqueeze.default(primals_294, -1)
        primals_294 = None
        unsqueeze_default_299 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_298, -1
        )
        unsqueeze_default_298 = None
        mul_tensor_524 = torch.ops.aten.mul.Tensor(
            mul_tensor_518, unsqueeze_default_297
        )
        mul_tensor_518 = unsqueeze_default_297 = None
        add_tensor_330 = torch.ops.aten.add.Tensor(
            mul_tensor_524, unsqueeze_default_299
        )
        mul_tensor_524 = unsqueeze_default_299 = None
        relu_default_74 = torch.ops.aten.relu.default(add_tensor_330)
        add_tensor_330 = None
        convolution_default_138 = torch.ops.aten.convolution.default(
            relu_default_74,
            primals_295,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_139 = torch.ops.aten.convolution.default(
            convolution_default_138,
            primals_296,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_150 = torch.ops.aten.var.correction(
            convolution_default_139, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_75 = torch.ops.aten.mean.dim(convolution_default_139, [0, 2, 3], True)
        add_tensor_331 = torch.ops.aten.add.Tensor(var_correction_150, 0.001)
        var_correction_150 = None
        sqrt_default_75 = torch.ops.aten.sqrt.default(add_tensor_331)
        add_tensor_331 = None
        reciprocal_default_75 = torch.ops.aten.reciprocal.default(sqrt_default_75)
        sqrt_default_75 = None
        sub_tensor_75 = torch.ops.aten.sub.Tensor(convolution_default_139, mean_dim_75)
        mul_tensor_525 = torch.ops.aten.mul.Tensor(sub_tensor_75, reciprocal_default_75)
        sub_tensor_75 = None
        squeeze_dim_450 = torch.ops.aten.squeeze.dim(mean_dim_75, 3)
        mean_dim_75 = None
        squeeze_dim_451 = torch.ops.aten.squeeze.dim(squeeze_dim_450, 2)
        squeeze_dim_450 = None
        squeeze_dim_452 = torch.ops.aten.squeeze.dim(squeeze_dim_451, 0)
        squeeze_dim_451 = None
        squeeze_dim_453 = torch.ops.aten.squeeze.dim(reciprocal_default_75, 3)
        reciprocal_default_75 = None
        squeeze_dim_454 = torch.ops.aten.squeeze.dim(squeeze_dim_453, 2)
        squeeze_dim_453 = None
        squeeze_dim_455 = torch.ops.aten.squeeze.dim(squeeze_dim_454, 0)
        squeeze_dim_454 = None
        unsqueeze_default_300 = torch.ops.aten.unsqueeze.default(primals_297, -1)
        unsqueeze_default_301 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_300, -1
        )
        unsqueeze_default_300 = None
        unsqueeze_default_302 = torch.ops.aten.unsqueeze.default(primals_298, -1)
        primals_298 = None
        unsqueeze_default_303 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_302, -1
        )
        unsqueeze_default_302 = None
        mul_tensor_531 = torch.ops.aten.mul.Tensor(
            mul_tensor_525, unsqueeze_default_301
        )
        mul_tensor_525 = unsqueeze_default_301 = None
        add_tensor_334 = torch.ops.aten.add.Tensor(
            mul_tensor_531, unsqueeze_default_303
        )
        mul_tensor_531 = unsqueeze_default_303 = None
        relu_default_75 = torch.ops.aten.relu.default(add_tensor_322)
        convolution_default_140 = torch.ops.aten.convolution.default(
            relu_default_75,
            primals_299,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_141 = torch.ops.aten.convolution.default(
            convolution_default_140,
            primals_300,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_152 = torch.ops.aten.var.correction(
            convolution_default_141, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_76 = torch.ops.aten.mean.dim(convolution_default_141, [0, 2, 3], True)
        add_tensor_335 = torch.ops.aten.add.Tensor(var_correction_152, 0.001)
        var_correction_152 = None
        sqrt_default_76 = torch.ops.aten.sqrt.default(add_tensor_335)
        add_tensor_335 = None
        reciprocal_default_76 = torch.ops.aten.reciprocal.default(sqrt_default_76)
        sqrt_default_76 = None
        sub_tensor_76 = torch.ops.aten.sub.Tensor(convolution_default_141, mean_dim_76)
        mul_tensor_532 = torch.ops.aten.mul.Tensor(sub_tensor_76, reciprocal_default_76)
        sub_tensor_76 = None
        squeeze_dim_456 = torch.ops.aten.squeeze.dim(mean_dim_76, 3)
        mean_dim_76 = None
        squeeze_dim_457 = torch.ops.aten.squeeze.dim(squeeze_dim_456, 2)
        squeeze_dim_456 = None
        squeeze_dim_458 = torch.ops.aten.squeeze.dim(squeeze_dim_457, 0)
        squeeze_dim_457 = None
        squeeze_dim_459 = torch.ops.aten.squeeze.dim(reciprocal_default_76, 3)
        reciprocal_default_76 = None
        squeeze_dim_460 = torch.ops.aten.squeeze.dim(squeeze_dim_459, 2)
        squeeze_dim_459 = None
        squeeze_dim_461 = torch.ops.aten.squeeze.dim(squeeze_dim_460, 0)
        squeeze_dim_460 = None
        unsqueeze_default_304 = torch.ops.aten.unsqueeze.default(primals_301, -1)
        unsqueeze_default_305 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_304, -1
        )
        unsqueeze_default_304 = None
        unsqueeze_default_306 = torch.ops.aten.unsqueeze.default(primals_302, -1)
        primals_302 = None
        unsqueeze_default_307 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_306, -1
        )
        unsqueeze_default_306 = None
        mul_tensor_538 = torch.ops.aten.mul.Tensor(
            mul_tensor_532, unsqueeze_default_305
        )
        mul_tensor_532 = unsqueeze_default_305 = None
        add_tensor_338 = torch.ops.aten.add.Tensor(
            mul_tensor_538, unsqueeze_default_307
        )
        mul_tensor_538 = unsqueeze_default_307 = None
        relu_default_76 = torch.ops.aten.relu.default(add_tensor_338)
        add_tensor_338 = None
        convolution_default_142 = torch.ops.aten.convolution.default(
            relu_default_76,
            primals_303,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_143 = torch.ops.aten.convolution.default(
            convolution_default_142,
            primals_304,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_154 = torch.ops.aten.var.correction(
            convolution_default_143, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_77 = torch.ops.aten.mean.dim(convolution_default_143, [0, 2, 3], True)
        add_tensor_339 = torch.ops.aten.add.Tensor(var_correction_154, 0.001)
        var_correction_154 = None
        sqrt_default_77 = torch.ops.aten.sqrt.default(add_tensor_339)
        add_tensor_339 = None
        reciprocal_default_77 = torch.ops.aten.reciprocal.default(sqrt_default_77)
        sqrt_default_77 = None
        sub_tensor_77 = torch.ops.aten.sub.Tensor(convolution_default_143, mean_dim_77)
        mul_tensor_539 = torch.ops.aten.mul.Tensor(sub_tensor_77, reciprocal_default_77)
        sub_tensor_77 = None
        squeeze_dim_462 = torch.ops.aten.squeeze.dim(mean_dim_77, 3)
        mean_dim_77 = None
        squeeze_dim_463 = torch.ops.aten.squeeze.dim(squeeze_dim_462, 2)
        squeeze_dim_462 = None
        squeeze_dim_464 = torch.ops.aten.squeeze.dim(squeeze_dim_463, 0)
        squeeze_dim_463 = None
        squeeze_dim_465 = torch.ops.aten.squeeze.dim(reciprocal_default_77, 3)
        reciprocal_default_77 = None
        squeeze_dim_466 = torch.ops.aten.squeeze.dim(squeeze_dim_465, 2)
        squeeze_dim_465 = None
        squeeze_dim_467 = torch.ops.aten.squeeze.dim(squeeze_dim_466, 0)
        squeeze_dim_466 = None
        unsqueeze_default_308 = torch.ops.aten.unsqueeze.default(primals_305, -1)
        unsqueeze_default_309 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_308, -1
        )
        unsqueeze_default_308 = None
        unsqueeze_default_310 = torch.ops.aten.unsqueeze.default(primals_306, -1)
        primals_306 = None
        unsqueeze_default_311 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_310, -1
        )
        unsqueeze_default_310 = None
        mul_tensor_545 = torch.ops.aten.mul.Tensor(
            mul_tensor_539, unsqueeze_default_309
        )
        mul_tensor_539 = unsqueeze_default_309 = None
        add_tensor_342 = torch.ops.aten.add.Tensor(
            mul_tensor_545, unsqueeze_default_311
        )
        mul_tensor_545 = unsqueeze_default_311 = None
        add_tensor_343 = torch.ops.aten.add.Tensor(add_tensor_334, add_tensor_342)
        add_tensor_334 = add_tensor_342 = None
        convolution_default_144 = torch.ops.aten.convolution.default(
            relu_default_75,
            primals_307,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_145 = torch.ops.aten.convolution.default(
            convolution_default_144,
            primals_308,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_156 = torch.ops.aten.var.correction(
            convolution_default_145, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_78 = torch.ops.aten.mean.dim(convolution_default_145, [0, 2, 3], True)
        add_tensor_344 = torch.ops.aten.add.Tensor(var_correction_156, 0.001)
        var_correction_156 = None
        sqrt_default_78 = torch.ops.aten.sqrt.default(add_tensor_344)
        add_tensor_344 = None
        reciprocal_default_78 = torch.ops.aten.reciprocal.default(sqrt_default_78)
        sqrt_default_78 = None
        sub_tensor_78 = torch.ops.aten.sub.Tensor(convolution_default_145, mean_dim_78)
        mul_tensor_546 = torch.ops.aten.mul.Tensor(sub_tensor_78, reciprocal_default_78)
        sub_tensor_78 = None
        squeeze_dim_468 = torch.ops.aten.squeeze.dim(mean_dim_78, 3)
        mean_dim_78 = None
        squeeze_dim_469 = torch.ops.aten.squeeze.dim(squeeze_dim_468, 2)
        squeeze_dim_468 = None
        squeeze_dim_470 = torch.ops.aten.squeeze.dim(squeeze_dim_469, 0)
        squeeze_dim_469 = None
        squeeze_dim_471 = torch.ops.aten.squeeze.dim(reciprocal_default_78, 3)
        reciprocal_default_78 = None
        squeeze_dim_472 = torch.ops.aten.squeeze.dim(squeeze_dim_471, 2)
        squeeze_dim_471 = None
        squeeze_dim_473 = torch.ops.aten.squeeze.dim(squeeze_dim_472, 0)
        squeeze_dim_472 = None
        unsqueeze_default_312 = torch.ops.aten.unsqueeze.default(primals_309, -1)
        unsqueeze_default_313 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_312, -1
        )
        unsqueeze_default_312 = None
        unsqueeze_default_314 = torch.ops.aten.unsqueeze.default(primals_310, -1)
        primals_310 = None
        unsqueeze_default_315 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_314, -1
        )
        unsqueeze_default_314 = None
        mul_tensor_552 = torch.ops.aten.mul.Tensor(
            mul_tensor_546, unsqueeze_default_313
        )
        mul_tensor_546 = unsqueeze_default_313 = None
        add_tensor_347 = torch.ops.aten.add.Tensor(
            mul_tensor_552, unsqueeze_default_315
        )
        mul_tensor_552 = unsqueeze_default_315 = None
        relu_default_78 = torch.ops.aten.relu.default(add_tensor_347)
        add_tensor_347 = None
        convolution_default_146 = torch.ops.aten.convolution.default(
            relu_default_78,
            primals_311,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_147 = torch.ops.aten.convolution.default(
            convolution_default_146,
            primals_312,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_158 = torch.ops.aten.var.correction(
            convolution_default_147, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_79 = torch.ops.aten.mean.dim(convolution_default_147, [0, 2, 3], True)
        add_tensor_348 = torch.ops.aten.add.Tensor(var_correction_158, 0.001)
        var_correction_158 = None
        sqrt_default_79 = torch.ops.aten.sqrt.default(add_tensor_348)
        add_tensor_348 = None
        reciprocal_default_79 = torch.ops.aten.reciprocal.default(sqrt_default_79)
        sqrt_default_79 = None
        sub_tensor_79 = torch.ops.aten.sub.Tensor(convolution_default_147, mean_dim_79)
        mul_tensor_553 = torch.ops.aten.mul.Tensor(sub_tensor_79, reciprocal_default_79)
        sub_tensor_79 = None
        squeeze_dim_474 = torch.ops.aten.squeeze.dim(mean_dim_79, 3)
        mean_dim_79 = None
        squeeze_dim_475 = torch.ops.aten.squeeze.dim(squeeze_dim_474, 2)
        squeeze_dim_474 = None
        squeeze_dim_476 = torch.ops.aten.squeeze.dim(squeeze_dim_475, 0)
        squeeze_dim_475 = None
        squeeze_dim_477 = torch.ops.aten.squeeze.dim(reciprocal_default_79, 3)
        reciprocal_default_79 = None
        squeeze_dim_478 = torch.ops.aten.squeeze.dim(squeeze_dim_477, 2)
        squeeze_dim_477 = None
        squeeze_dim_479 = torch.ops.aten.squeeze.dim(squeeze_dim_478, 0)
        squeeze_dim_478 = None
        unsqueeze_default_316 = torch.ops.aten.unsqueeze.default(primals_313, -1)
        unsqueeze_default_317 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_316, -1
        )
        unsqueeze_default_316 = None
        unsqueeze_default_318 = torch.ops.aten.unsqueeze.default(primals_314, -1)
        primals_314 = None
        unsqueeze_default_319 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_318, -1
        )
        unsqueeze_default_318 = None
        mul_tensor_559 = torch.ops.aten.mul.Tensor(
            mul_tensor_553, unsqueeze_default_317
        )
        mul_tensor_553 = unsqueeze_default_317 = None
        add_tensor_351 = torch.ops.aten.add.Tensor(
            mul_tensor_559, unsqueeze_default_319
        )
        mul_tensor_559 = unsqueeze_default_319 = None
        convolution_default_148 = torch.ops.aten.convolution.default(
            relu_default_75,
            primals_315,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_75 = None
        convolution_default_149 = torch.ops.aten.convolution.default(
            convolution_default_148,
            primals_316,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_160 = torch.ops.aten.var.correction(
            convolution_default_149, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_80 = torch.ops.aten.mean.dim(convolution_default_149, [0, 2, 3], True)
        add_tensor_352 = torch.ops.aten.add.Tensor(var_correction_160, 0.001)
        var_correction_160 = None
        sqrt_default_80 = torch.ops.aten.sqrt.default(add_tensor_352)
        add_tensor_352 = None
        reciprocal_default_80 = torch.ops.aten.reciprocal.default(sqrt_default_80)
        sqrt_default_80 = None
        sub_tensor_80 = torch.ops.aten.sub.Tensor(convolution_default_149, mean_dim_80)
        mul_tensor_560 = torch.ops.aten.mul.Tensor(sub_tensor_80, reciprocal_default_80)
        sub_tensor_80 = None
        squeeze_dim_480 = torch.ops.aten.squeeze.dim(mean_dim_80, 3)
        mean_dim_80 = None
        squeeze_dim_481 = torch.ops.aten.squeeze.dim(squeeze_dim_480, 2)
        squeeze_dim_480 = None
        squeeze_dim_482 = torch.ops.aten.squeeze.dim(squeeze_dim_481, 0)
        squeeze_dim_481 = None
        squeeze_dim_483 = torch.ops.aten.squeeze.dim(reciprocal_default_80, 3)
        reciprocal_default_80 = None
        squeeze_dim_484 = torch.ops.aten.squeeze.dim(squeeze_dim_483, 2)
        squeeze_dim_483 = None
        squeeze_dim_485 = torch.ops.aten.squeeze.dim(squeeze_dim_484, 0)
        squeeze_dim_484 = None
        unsqueeze_default_320 = torch.ops.aten.unsqueeze.default(primals_317, -1)
        unsqueeze_default_321 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_320, -1
        )
        unsqueeze_default_320 = None
        unsqueeze_default_322 = torch.ops.aten.unsqueeze.default(primals_318, -1)
        primals_318 = None
        unsqueeze_default_323 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_322, -1
        )
        unsqueeze_default_322 = None
        mul_tensor_566 = torch.ops.aten.mul.Tensor(
            mul_tensor_560, unsqueeze_default_321
        )
        mul_tensor_560 = unsqueeze_default_321 = None
        add_tensor_355 = torch.ops.aten.add.Tensor(
            mul_tensor_566, unsqueeze_default_323
        )
        mul_tensor_566 = unsqueeze_default_323 = None
        relu_default_80 = torch.ops.aten.relu.default(add_tensor_355)
        add_tensor_355 = None
        convolution_default_150 = torch.ops.aten.convolution.default(
            relu_default_80,
            primals_319,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_151 = torch.ops.aten.convolution.default(
            convolution_default_150,
            primals_320,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_162 = torch.ops.aten.var.correction(
            convolution_default_151, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_81 = torch.ops.aten.mean.dim(convolution_default_151, [0, 2, 3], True)
        add_tensor_356 = torch.ops.aten.add.Tensor(var_correction_162, 0.001)
        var_correction_162 = None
        sqrt_default_81 = torch.ops.aten.sqrt.default(add_tensor_356)
        add_tensor_356 = None
        reciprocal_default_81 = torch.ops.aten.reciprocal.default(sqrt_default_81)
        sqrt_default_81 = None
        sub_tensor_81 = torch.ops.aten.sub.Tensor(convolution_default_151, mean_dim_81)
        mul_tensor_567 = torch.ops.aten.mul.Tensor(sub_tensor_81, reciprocal_default_81)
        sub_tensor_81 = None
        squeeze_dim_486 = torch.ops.aten.squeeze.dim(mean_dim_81, 3)
        mean_dim_81 = None
        squeeze_dim_487 = torch.ops.aten.squeeze.dim(squeeze_dim_486, 2)
        squeeze_dim_486 = None
        squeeze_dim_488 = torch.ops.aten.squeeze.dim(squeeze_dim_487, 0)
        squeeze_dim_487 = None
        squeeze_dim_489 = torch.ops.aten.squeeze.dim(reciprocal_default_81, 3)
        reciprocal_default_81 = None
        squeeze_dim_490 = torch.ops.aten.squeeze.dim(squeeze_dim_489, 2)
        squeeze_dim_489 = None
        squeeze_dim_491 = torch.ops.aten.squeeze.dim(squeeze_dim_490, 0)
        squeeze_dim_490 = None
        unsqueeze_default_324 = torch.ops.aten.unsqueeze.default(primals_321, -1)
        unsqueeze_default_325 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_324, -1
        )
        unsqueeze_default_324 = None
        unsqueeze_default_326 = torch.ops.aten.unsqueeze.default(primals_322, -1)
        primals_322 = None
        unsqueeze_default_327 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_326, -1
        )
        unsqueeze_default_326 = None
        mul_tensor_573 = torch.ops.aten.mul.Tensor(
            mul_tensor_567, unsqueeze_default_325
        )
        mul_tensor_567 = unsqueeze_default_325 = None
        add_tensor_359 = torch.ops.aten.add.Tensor(
            mul_tensor_573, unsqueeze_default_327
        )
        mul_tensor_573 = unsqueeze_default_327 = None
        add_tensor_360 = torch.ops.aten.add.Tensor(add_tensor_351, add_tensor_359)
        add_tensor_351 = add_tensor_359 = None
        avg_pool2d_default_20 = torch.ops.aten.avg_pool2d.default(
            add_tensor_326, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_361 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_20, add_tensor_322
        )
        avg_pool2d_default_20 = None
        avg_pool2d_default_21 = torch.ops.aten.avg_pool2d.default(
            add_tensor_322, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_362 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_21, avg_pool2d_default_21
        )
        avg_pool2d_default_21 = None
        convolution_default_152 = torch.ops.aten.convolution.default(
            relu_default_73,
            primals_323,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_73 = None
        convolution_default_153 = torch.ops.aten.convolution.default(
            convolution_default_152,
            primals_324,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_164 = torch.ops.aten.var.correction(
            convolution_default_153, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_82 = torch.ops.aten.mean.dim(convolution_default_153, [0, 2, 3], True)
        add_tensor_363 = torch.ops.aten.add.Tensor(var_correction_164, 0.001)
        var_correction_164 = None
        sqrt_default_82 = torch.ops.aten.sqrt.default(add_tensor_363)
        add_tensor_363 = None
        reciprocal_default_82 = torch.ops.aten.reciprocal.default(sqrt_default_82)
        sqrt_default_82 = None
        sub_tensor_82 = torch.ops.aten.sub.Tensor(convolution_default_153, mean_dim_82)
        mul_tensor_574 = torch.ops.aten.mul.Tensor(sub_tensor_82, reciprocal_default_82)
        sub_tensor_82 = None
        squeeze_dim_492 = torch.ops.aten.squeeze.dim(mean_dim_82, 3)
        mean_dim_82 = None
        squeeze_dim_493 = torch.ops.aten.squeeze.dim(squeeze_dim_492, 2)
        squeeze_dim_492 = None
        squeeze_dim_494 = torch.ops.aten.squeeze.dim(squeeze_dim_493, 0)
        squeeze_dim_493 = None
        squeeze_dim_495 = torch.ops.aten.squeeze.dim(reciprocal_default_82, 3)
        reciprocal_default_82 = None
        squeeze_dim_496 = torch.ops.aten.squeeze.dim(squeeze_dim_495, 2)
        squeeze_dim_495 = None
        squeeze_dim_497 = torch.ops.aten.squeeze.dim(squeeze_dim_496, 0)
        squeeze_dim_496 = None
        unsqueeze_default_328 = torch.ops.aten.unsqueeze.default(primals_325, -1)
        unsqueeze_default_329 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_328, -1
        )
        unsqueeze_default_328 = None
        unsqueeze_default_330 = torch.ops.aten.unsqueeze.default(primals_326, -1)
        primals_326 = None
        unsqueeze_default_331 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_330, -1
        )
        unsqueeze_default_330 = None
        mul_tensor_580 = torch.ops.aten.mul.Tensor(
            mul_tensor_574, unsqueeze_default_329
        )
        mul_tensor_574 = unsqueeze_default_329 = None
        add_tensor_366 = torch.ops.aten.add.Tensor(
            mul_tensor_580, unsqueeze_default_331
        )
        mul_tensor_580 = unsqueeze_default_331 = None
        relu_default_82 = torch.ops.aten.relu.default(add_tensor_366)
        add_tensor_366 = None
        convolution_default_154 = torch.ops.aten.convolution.default(
            relu_default_82,
            primals_327,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_155 = torch.ops.aten.convolution.default(
            convolution_default_154,
            primals_328,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_166 = torch.ops.aten.var.correction(
            convolution_default_155, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_83 = torch.ops.aten.mean.dim(convolution_default_155, [0, 2, 3], True)
        add_tensor_367 = torch.ops.aten.add.Tensor(var_correction_166, 0.001)
        var_correction_166 = None
        sqrt_default_83 = torch.ops.aten.sqrt.default(add_tensor_367)
        add_tensor_367 = None
        reciprocal_default_83 = torch.ops.aten.reciprocal.default(sqrt_default_83)
        sqrt_default_83 = None
        sub_tensor_83 = torch.ops.aten.sub.Tensor(convolution_default_155, mean_dim_83)
        mul_tensor_581 = torch.ops.aten.mul.Tensor(sub_tensor_83, reciprocal_default_83)
        sub_tensor_83 = None
        squeeze_dim_498 = torch.ops.aten.squeeze.dim(mean_dim_83, 3)
        mean_dim_83 = None
        squeeze_dim_499 = torch.ops.aten.squeeze.dim(squeeze_dim_498, 2)
        squeeze_dim_498 = None
        squeeze_dim_500 = torch.ops.aten.squeeze.dim(squeeze_dim_499, 0)
        squeeze_dim_499 = None
        squeeze_dim_501 = torch.ops.aten.squeeze.dim(reciprocal_default_83, 3)
        reciprocal_default_83 = None
        squeeze_dim_502 = torch.ops.aten.squeeze.dim(squeeze_dim_501, 2)
        squeeze_dim_501 = None
        squeeze_dim_503 = torch.ops.aten.squeeze.dim(squeeze_dim_502, 0)
        squeeze_dim_502 = None
        unsqueeze_default_332 = torch.ops.aten.unsqueeze.default(primals_329, -1)
        unsqueeze_default_333 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_332, -1
        )
        unsqueeze_default_332 = None
        unsqueeze_default_334 = torch.ops.aten.unsqueeze.default(primals_330, -1)
        primals_330 = None
        unsqueeze_default_335 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_334, -1
        )
        unsqueeze_default_334 = None
        mul_tensor_587 = torch.ops.aten.mul.Tensor(
            mul_tensor_581, unsqueeze_default_333
        )
        mul_tensor_581 = unsqueeze_default_333 = None
        add_tensor_370 = torch.ops.aten.add.Tensor(
            mul_tensor_587, unsqueeze_default_335
        )
        mul_tensor_587 = unsqueeze_default_335 = None
        add_tensor_371 = torch.ops.aten.add.Tensor(add_tensor_370, add_tensor_326)
        add_tensor_370 = add_tensor_326 = None
        cat_default_8 = torch.ops.aten.cat.default(
            [
                add_tensor_322,
                add_tensor_343,
                add_tensor_360,
                add_tensor_361,
                add_tensor_362,
                add_tensor_371,
            ],
            1,
        )
        add_tensor_322 = (
            add_tensor_343
        ) = add_tensor_360 = add_tensor_361 = add_tensor_362 = add_tensor_371 = None
        convolution_default_156 = torch.ops.aten.convolution.default(
            relu_default_72, primals_331, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_168 = torch.ops.aten.var.correction(
            convolution_default_156, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_84 = torch.ops.aten.mean.dim(convolution_default_156, [0, 2, 3], True)
        add_tensor_372 = torch.ops.aten.add.Tensor(var_correction_168, 0.001)
        var_correction_168 = None
        sqrt_default_84 = torch.ops.aten.sqrt.default(add_tensor_372)
        add_tensor_372 = None
        reciprocal_default_84 = torch.ops.aten.reciprocal.default(sqrt_default_84)
        sqrt_default_84 = None
        sub_tensor_84 = torch.ops.aten.sub.Tensor(convolution_default_156, mean_dim_84)
        mul_tensor_588 = torch.ops.aten.mul.Tensor(sub_tensor_84, reciprocal_default_84)
        sub_tensor_84 = None
        unsqueeze_default_336 = torch.ops.aten.unsqueeze.default(primals_332, -1)
        unsqueeze_default_337 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_336, -1
        )
        unsqueeze_default_336 = None
        unsqueeze_default_338 = torch.ops.aten.unsqueeze.default(primals_333, -1)
        unsqueeze_default_339 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_338, -1
        )
        unsqueeze_default_338 = None
        mul_tensor_594 = torch.ops.aten.mul.Tensor(
            mul_tensor_588, unsqueeze_default_337
        )
        mul_tensor_588 = unsqueeze_default_337 = None
        add_tensor_375 = torch.ops.aten.add.Tensor(
            mul_tensor_594, unsqueeze_default_339
        )
        mul_tensor_594 = unsqueeze_default_339 = None
        relu_default_84 = torch.ops.aten.relu.default(cat_default_8)
        cat_default_8 = None
        convolution_default_157 = torch.ops.aten.convolution.default(
            relu_default_84, primals_334, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_170 = torch.ops.aten.var.correction(
            convolution_default_157, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_85 = torch.ops.aten.mean.dim(convolution_default_157, [0, 2, 3], True)
        add_tensor_376 = torch.ops.aten.add.Tensor(var_correction_170, 0.001)
        var_correction_170 = None
        sqrt_default_85 = torch.ops.aten.sqrt.default(add_tensor_376)
        add_tensor_376 = None
        reciprocal_default_85 = torch.ops.aten.reciprocal.default(sqrt_default_85)
        sqrt_default_85 = None
        sub_tensor_85 = torch.ops.aten.sub.Tensor(convolution_default_157, mean_dim_85)
        mul_tensor_595 = torch.ops.aten.mul.Tensor(sub_tensor_85, reciprocal_default_85)
        sub_tensor_85 = None
        unsqueeze_default_340 = torch.ops.aten.unsqueeze.default(primals_335, -1)
        unsqueeze_default_341 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_340, -1
        )
        unsqueeze_default_340 = None
        unsqueeze_default_342 = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_default_343 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_342, -1
        )
        unsqueeze_default_342 = None
        mul_tensor_601 = torch.ops.aten.mul.Tensor(
            mul_tensor_595, unsqueeze_default_341
        )
        mul_tensor_595 = unsqueeze_default_341 = None
        add_tensor_379 = torch.ops.aten.add.Tensor(
            mul_tensor_601, unsqueeze_default_343
        )
        mul_tensor_601 = unsqueeze_default_343 = None
        relu_default_85 = torch.ops.aten.relu.default(add_tensor_379)
        convolution_default_158 = torch.ops.aten.convolution.default(
            relu_default_85,
            primals_337,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_159 = torch.ops.aten.convolution.default(
            convolution_default_158,
            primals_338,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_172 = torch.ops.aten.var.correction(
            convolution_default_159, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_86 = torch.ops.aten.mean.dim(convolution_default_159, [0, 2, 3], True)
        add_tensor_380 = torch.ops.aten.add.Tensor(var_correction_172, 0.001)
        var_correction_172 = None
        sqrt_default_86 = torch.ops.aten.sqrt.default(add_tensor_380)
        add_tensor_380 = None
        reciprocal_default_86 = torch.ops.aten.reciprocal.default(sqrt_default_86)
        sqrt_default_86 = None
        sub_tensor_86 = torch.ops.aten.sub.Tensor(convolution_default_159, mean_dim_86)
        mul_tensor_602 = torch.ops.aten.mul.Tensor(sub_tensor_86, reciprocal_default_86)
        sub_tensor_86 = None
        squeeze_dim_516 = torch.ops.aten.squeeze.dim(mean_dim_86, 3)
        mean_dim_86 = None
        squeeze_dim_517 = torch.ops.aten.squeeze.dim(squeeze_dim_516, 2)
        squeeze_dim_516 = None
        squeeze_dim_518 = torch.ops.aten.squeeze.dim(squeeze_dim_517, 0)
        squeeze_dim_517 = None
        squeeze_dim_519 = torch.ops.aten.squeeze.dim(reciprocal_default_86, 3)
        reciprocal_default_86 = None
        squeeze_dim_520 = torch.ops.aten.squeeze.dim(squeeze_dim_519, 2)
        squeeze_dim_519 = None
        squeeze_dim_521 = torch.ops.aten.squeeze.dim(squeeze_dim_520, 0)
        squeeze_dim_520 = None
        unsqueeze_default_344 = torch.ops.aten.unsqueeze.default(primals_339, -1)
        unsqueeze_default_345 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_344, -1
        )
        unsqueeze_default_344 = None
        unsqueeze_default_346 = torch.ops.aten.unsqueeze.default(primals_340, -1)
        primals_340 = None
        unsqueeze_default_347 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_346, -1
        )
        unsqueeze_default_346 = None
        mul_tensor_608 = torch.ops.aten.mul.Tensor(
            mul_tensor_602, unsqueeze_default_345
        )
        mul_tensor_602 = unsqueeze_default_345 = None
        add_tensor_383 = torch.ops.aten.add.Tensor(
            mul_tensor_608, unsqueeze_default_347
        )
        mul_tensor_608 = unsqueeze_default_347 = None
        relu_default_86 = torch.ops.aten.relu.default(add_tensor_383)
        add_tensor_383 = None
        convolution_default_160 = torch.ops.aten.convolution.default(
            relu_default_86,
            primals_341,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_161 = torch.ops.aten.convolution.default(
            convolution_default_160,
            primals_342,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_174 = torch.ops.aten.var.correction(
            convolution_default_161, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_87 = torch.ops.aten.mean.dim(convolution_default_161, [0, 2, 3], True)
        add_tensor_384 = torch.ops.aten.add.Tensor(var_correction_174, 0.001)
        var_correction_174 = None
        sqrt_default_87 = torch.ops.aten.sqrt.default(add_tensor_384)
        add_tensor_384 = None
        reciprocal_default_87 = torch.ops.aten.reciprocal.default(sqrt_default_87)
        sqrt_default_87 = None
        sub_tensor_87 = torch.ops.aten.sub.Tensor(convolution_default_161, mean_dim_87)
        mul_tensor_609 = torch.ops.aten.mul.Tensor(sub_tensor_87, reciprocal_default_87)
        sub_tensor_87 = None
        squeeze_dim_522 = torch.ops.aten.squeeze.dim(mean_dim_87, 3)
        mean_dim_87 = None
        squeeze_dim_523 = torch.ops.aten.squeeze.dim(squeeze_dim_522, 2)
        squeeze_dim_522 = None
        squeeze_dim_524 = torch.ops.aten.squeeze.dim(squeeze_dim_523, 0)
        squeeze_dim_523 = None
        squeeze_dim_525 = torch.ops.aten.squeeze.dim(reciprocal_default_87, 3)
        reciprocal_default_87 = None
        squeeze_dim_526 = torch.ops.aten.squeeze.dim(squeeze_dim_525, 2)
        squeeze_dim_525 = None
        squeeze_dim_527 = torch.ops.aten.squeeze.dim(squeeze_dim_526, 0)
        squeeze_dim_526 = None
        unsqueeze_default_348 = torch.ops.aten.unsqueeze.default(primals_343, -1)
        unsqueeze_default_349 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_348, -1
        )
        unsqueeze_default_348 = None
        unsqueeze_default_350 = torch.ops.aten.unsqueeze.default(primals_344, -1)
        primals_344 = None
        unsqueeze_default_351 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_350, -1
        )
        unsqueeze_default_350 = None
        mul_tensor_615 = torch.ops.aten.mul.Tensor(
            mul_tensor_609, unsqueeze_default_349
        )
        mul_tensor_609 = unsqueeze_default_349 = None
        add_tensor_387 = torch.ops.aten.add.Tensor(
            mul_tensor_615, unsqueeze_default_351
        )
        mul_tensor_615 = unsqueeze_default_351 = None
        relu_default_87 = torch.ops.aten.relu.default(add_tensor_375)
        convolution_default_162 = torch.ops.aten.convolution.default(
            relu_default_87,
            primals_345,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_163 = torch.ops.aten.convolution.default(
            convolution_default_162,
            primals_346,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_176 = torch.ops.aten.var.correction(
            convolution_default_163, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_88 = torch.ops.aten.mean.dim(convolution_default_163, [0, 2, 3], True)
        add_tensor_388 = torch.ops.aten.add.Tensor(var_correction_176, 0.001)
        var_correction_176 = None
        sqrt_default_88 = torch.ops.aten.sqrt.default(add_tensor_388)
        add_tensor_388 = None
        reciprocal_default_88 = torch.ops.aten.reciprocal.default(sqrt_default_88)
        sqrt_default_88 = None
        sub_tensor_88 = torch.ops.aten.sub.Tensor(convolution_default_163, mean_dim_88)
        mul_tensor_616 = torch.ops.aten.mul.Tensor(sub_tensor_88, reciprocal_default_88)
        sub_tensor_88 = None
        squeeze_dim_528 = torch.ops.aten.squeeze.dim(mean_dim_88, 3)
        mean_dim_88 = None
        squeeze_dim_529 = torch.ops.aten.squeeze.dim(squeeze_dim_528, 2)
        squeeze_dim_528 = None
        squeeze_dim_530 = torch.ops.aten.squeeze.dim(squeeze_dim_529, 0)
        squeeze_dim_529 = None
        squeeze_dim_531 = torch.ops.aten.squeeze.dim(reciprocal_default_88, 3)
        reciprocal_default_88 = None
        squeeze_dim_532 = torch.ops.aten.squeeze.dim(squeeze_dim_531, 2)
        squeeze_dim_531 = None
        squeeze_dim_533 = torch.ops.aten.squeeze.dim(squeeze_dim_532, 0)
        squeeze_dim_532 = None
        unsqueeze_default_352 = torch.ops.aten.unsqueeze.default(primals_347, -1)
        unsqueeze_default_353 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_352, -1
        )
        unsqueeze_default_352 = None
        unsqueeze_default_354 = torch.ops.aten.unsqueeze.default(primals_348, -1)
        primals_348 = None
        unsqueeze_default_355 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_354, -1
        )
        unsqueeze_default_354 = None
        mul_tensor_622 = torch.ops.aten.mul.Tensor(
            mul_tensor_616, unsqueeze_default_353
        )
        mul_tensor_616 = unsqueeze_default_353 = None
        add_tensor_391 = torch.ops.aten.add.Tensor(
            mul_tensor_622, unsqueeze_default_355
        )
        mul_tensor_622 = unsqueeze_default_355 = None
        relu_default_88 = torch.ops.aten.relu.default(add_tensor_391)
        add_tensor_391 = None
        convolution_default_164 = torch.ops.aten.convolution.default(
            relu_default_88,
            primals_349,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_165 = torch.ops.aten.convolution.default(
            convolution_default_164,
            primals_350,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_178 = torch.ops.aten.var.correction(
            convolution_default_165, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_89 = torch.ops.aten.mean.dim(convolution_default_165, [0, 2, 3], True)
        add_tensor_392 = torch.ops.aten.add.Tensor(var_correction_178, 0.001)
        var_correction_178 = None
        sqrt_default_89 = torch.ops.aten.sqrt.default(add_tensor_392)
        add_tensor_392 = None
        reciprocal_default_89 = torch.ops.aten.reciprocal.default(sqrt_default_89)
        sqrt_default_89 = None
        sub_tensor_89 = torch.ops.aten.sub.Tensor(convolution_default_165, mean_dim_89)
        mul_tensor_623 = torch.ops.aten.mul.Tensor(sub_tensor_89, reciprocal_default_89)
        sub_tensor_89 = None
        squeeze_dim_534 = torch.ops.aten.squeeze.dim(mean_dim_89, 3)
        mean_dim_89 = None
        squeeze_dim_535 = torch.ops.aten.squeeze.dim(squeeze_dim_534, 2)
        squeeze_dim_534 = None
        squeeze_dim_536 = torch.ops.aten.squeeze.dim(squeeze_dim_535, 0)
        squeeze_dim_535 = None
        squeeze_dim_537 = torch.ops.aten.squeeze.dim(reciprocal_default_89, 3)
        reciprocal_default_89 = None
        squeeze_dim_538 = torch.ops.aten.squeeze.dim(squeeze_dim_537, 2)
        squeeze_dim_537 = None
        squeeze_dim_539 = torch.ops.aten.squeeze.dim(squeeze_dim_538, 0)
        squeeze_dim_538 = None
        unsqueeze_default_356 = torch.ops.aten.unsqueeze.default(primals_351, -1)
        unsqueeze_default_357 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_356, -1
        )
        unsqueeze_default_356 = None
        unsqueeze_default_358 = torch.ops.aten.unsqueeze.default(primals_352, -1)
        primals_352 = None
        unsqueeze_default_359 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_358, -1
        )
        unsqueeze_default_358 = None
        mul_tensor_629 = torch.ops.aten.mul.Tensor(
            mul_tensor_623, unsqueeze_default_357
        )
        mul_tensor_623 = unsqueeze_default_357 = None
        add_tensor_395 = torch.ops.aten.add.Tensor(
            mul_tensor_629, unsqueeze_default_359
        )
        mul_tensor_629 = unsqueeze_default_359 = None
        add_tensor_396 = torch.ops.aten.add.Tensor(add_tensor_387, add_tensor_395)
        add_tensor_387 = add_tensor_395 = None
        convolution_default_166 = torch.ops.aten.convolution.default(
            relu_default_87,
            primals_353,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_167 = torch.ops.aten.convolution.default(
            convolution_default_166,
            primals_354,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_180 = torch.ops.aten.var.correction(
            convolution_default_167, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_90 = torch.ops.aten.mean.dim(convolution_default_167, [0, 2, 3], True)
        add_tensor_397 = torch.ops.aten.add.Tensor(var_correction_180, 0.001)
        var_correction_180 = None
        sqrt_default_90 = torch.ops.aten.sqrt.default(add_tensor_397)
        add_tensor_397 = None
        reciprocal_default_90 = torch.ops.aten.reciprocal.default(sqrt_default_90)
        sqrt_default_90 = None
        sub_tensor_90 = torch.ops.aten.sub.Tensor(convolution_default_167, mean_dim_90)
        mul_tensor_630 = torch.ops.aten.mul.Tensor(sub_tensor_90, reciprocal_default_90)
        sub_tensor_90 = None
        squeeze_dim_540 = torch.ops.aten.squeeze.dim(mean_dim_90, 3)
        mean_dim_90 = None
        squeeze_dim_541 = torch.ops.aten.squeeze.dim(squeeze_dim_540, 2)
        squeeze_dim_540 = None
        squeeze_dim_542 = torch.ops.aten.squeeze.dim(squeeze_dim_541, 0)
        squeeze_dim_541 = None
        squeeze_dim_543 = torch.ops.aten.squeeze.dim(reciprocal_default_90, 3)
        reciprocal_default_90 = None
        squeeze_dim_544 = torch.ops.aten.squeeze.dim(squeeze_dim_543, 2)
        squeeze_dim_543 = None
        squeeze_dim_545 = torch.ops.aten.squeeze.dim(squeeze_dim_544, 0)
        squeeze_dim_544 = None
        unsqueeze_default_360 = torch.ops.aten.unsqueeze.default(primals_355, -1)
        unsqueeze_default_361 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_360, -1
        )
        unsqueeze_default_360 = None
        unsqueeze_default_362 = torch.ops.aten.unsqueeze.default(primals_356, -1)
        primals_356 = None
        unsqueeze_default_363 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_362, -1
        )
        unsqueeze_default_362 = None
        mul_tensor_636 = torch.ops.aten.mul.Tensor(
            mul_tensor_630, unsqueeze_default_361
        )
        mul_tensor_630 = unsqueeze_default_361 = None
        add_tensor_400 = torch.ops.aten.add.Tensor(
            mul_tensor_636, unsqueeze_default_363
        )
        mul_tensor_636 = unsqueeze_default_363 = None
        relu_default_90 = torch.ops.aten.relu.default(add_tensor_400)
        add_tensor_400 = None
        convolution_default_168 = torch.ops.aten.convolution.default(
            relu_default_90,
            primals_357,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_169 = torch.ops.aten.convolution.default(
            convolution_default_168,
            primals_358,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_182 = torch.ops.aten.var.correction(
            convolution_default_169, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_91 = torch.ops.aten.mean.dim(convolution_default_169, [0, 2, 3], True)
        add_tensor_401 = torch.ops.aten.add.Tensor(var_correction_182, 0.001)
        var_correction_182 = None
        sqrt_default_91 = torch.ops.aten.sqrt.default(add_tensor_401)
        add_tensor_401 = None
        reciprocal_default_91 = torch.ops.aten.reciprocal.default(sqrt_default_91)
        sqrt_default_91 = None
        sub_tensor_91 = torch.ops.aten.sub.Tensor(convolution_default_169, mean_dim_91)
        mul_tensor_637 = torch.ops.aten.mul.Tensor(sub_tensor_91, reciprocal_default_91)
        sub_tensor_91 = None
        squeeze_dim_546 = torch.ops.aten.squeeze.dim(mean_dim_91, 3)
        mean_dim_91 = None
        squeeze_dim_547 = torch.ops.aten.squeeze.dim(squeeze_dim_546, 2)
        squeeze_dim_546 = None
        squeeze_dim_548 = torch.ops.aten.squeeze.dim(squeeze_dim_547, 0)
        squeeze_dim_547 = None
        squeeze_dim_549 = torch.ops.aten.squeeze.dim(reciprocal_default_91, 3)
        reciprocal_default_91 = None
        squeeze_dim_550 = torch.ops.aten.squeeze.dim(squeeze_dim_549, 2)
        squeeze_dim_549 = None
        squeeze_dim_551 = torch.ops.aten.squeeze.dim(squeeze_dim_550, 0)
        squeeze_dim_550 = None
        unsqueeze_default_364 = torch.ops.aten.unsqueeze.default(primals_359, -1)
        unsqueeze_default_365 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_364, -1
        )
        unsqueeze_default_364 = None
        unsqueeze_default_366 = torch.ops.aten.unsqueeze.default(primals_360, -1)
        primals_360 = None
        unsqueeze_default_367 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_366, -1
        )
        unsqueeze_default_366 = None
        mul_tensor_643 = torch.ops.aten.mul.Tensor(
            mul_tensor_637, unsqueeze_default_365
        )
        mul_tensor_637 = unsqueeze_default_365 = None
        add_tensor_404 = torch.ops.aten.add.Tensor(
            mul_tensor_643, unsqueeze_default_367
        )
        mul_tensor_643 = unsqueeze_default_367 = None
        convolution_default_170 = torch.ops.aten.convolution.default(
            relu_default_87,
            primals_361,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_87 = None
        convolution_default_171 = torch.ops.aten.convolution.default(
            convolution_default_170,
            primals_362,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_184 = torch.ops.aten.var.correction(
            convolution_default_171, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_92 = torch.ops.aten.mean.dim(convolution_default_171, [0, 2, 3], True)
        add_tensor_405 = torch.ops.aten.add.Tensor(var_correction_184, 0.001)
        var_correction_184 = None
        sqrt_default_92 = torch.ops.aten.sqrt.default(add_tensor_405)
        add_tensor_405 = None
        reciprocal_default_92 = torch.ops.aten.reciprocal.default(sqrt_default_92)
        sqrt_default_92 = None
        sub_tensor_92 = torch.ops.aten.sub.Tensor(convolution_default_171, mean_dim_92)
        mul_tensor_644 = torch.ops.aten.mul.Tensor(sub_tensor_92, reciprocal_default_92)
        sub_tensor_92 = None
        squeeze_dim_552 = torch.ops.aten.squeeze.dim(mean_dim_92, 3)
        mean_dim_92 = None
        squeeze_dim_553 = torch.ops.aten.squeeze.dim(squeeze_dim_552, 2)
        squeeze_dim_552 = None
        squeeze_dim_554 = torch.ops.aten.squeeze.dim(squeeze_dim_553, 0)
        squeeze_dim_553 = None
        squeeze_dim_555 = torch.ops.aten.squeeze.dim(reciprocal_default_92, 3)
        reciprocal_default_92 = None
        squeeze_dim_556 = torch.ops.aten.squeeze.dim(squeeze_dim_555, 2)
        squeeze_dim_555 = None
        squeeze_dim_557 = torch.ops.aten.squeeze.dim(squeeze_dim_556, 0)
        squeeze_dim_556 = None
        unsqueeze_default_368 = torch.ops.aten.unsqueeze.default(primals_363, -1)
        unsqueeze_default_369 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_368, -1
        )
        unsqueeze_default_368 = None
        unsqueeze_default_370 = torch.ops.aten.unsqueeze.default(primals_364, -1)
        primals_364 = None
        unsqueeze_default_371 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_370, -1
        )
        unsqueeze_default_370 = None
        mul_tensor_650 = torch.ops.aten.mul.Tensor(
            mul_tensor_644, unsqueeze_default_369
        )
        mul_tensor_644 = unsqueeze_default_369 = None
        add_tensor_408 = torch.ops.aten.add.Tensor(
            mul_tensor_650, unsqueeze_default_371
        )
        mul_tensor_650 = unsqueeze_default_371 = None
        relu_default_92 = torch.ops.aten.relu.default(add_tensor_408)
        add_tensor_408 = None
        convolution_default_172 = torch.ops.aten.convolution.default(
            relu_default_92,
            primals_365,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_173 = torch.ops.aten.convolution.default(
            convolution_default_172,
            primals_366,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_186 = torch.ops.aten.var.correction(
            convolution_default_173, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_93 = torch.ops.aten.mean.dim(convolution_default_173, [0, 2, 3], True)
        add_tensor_409 = torch.ops.aten.add.Tensor(var_correction_186, 0.001)
        var_correction_186 = None
        sqrt_default_93 = torch.ops.aten.sqrt.default(add_tensor_409)
        add_tensor_409 = None
        reciprocal_default_93 = torch.ops.aten.reciprocal.default(sqrt_default_93)
        sqrt_default_93 = None
        sub_tensor_93 = torch.ops.aten.sub.Tensor(convolution_default_173, mean_dim_93)
        mul_tensor_651 = torch.ops.aten.mul.Tensor(sub_tensor_93, reciprocal_default_93)
        sub_tensor_93 = None
        squeeze_dim_558 = torch.ops.aten.squeeze.dim(mean_dim_93, 3)
        mean_dim_93 = None
        squeeze_dim_559 = torch.ops.aten.squeeze.dim(squeeze_dim_558, 2)
        squeeze_dim_558 = None
        squeeze_dim_560 = torch.ops.aten.squeeze.dim(squeeze_dim_559, 0)
        squeeze_dim_559 = None
        squeeze_dim_561 = torch.ops.aten.squeeze.dim(reciprocal_default_93, 3)
        reciprocal_default_93 = None
        squeeze_dim_562 = torch.ops.aten.squeeze.dim(squeeze_dim_561, 2)
        squeeze_dim_561 = None
        squeeze_dim_563 = torch.ops.aten.squeeze.dim(squeeze_dim_562, 0)
        squeeze_dim_562 = None
        unsqueeze_default_372 = torch.ops.aten.unsqueeze.default(primals_367, -1)
        unsqueeze_default_373 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_372, -1
        )
        unsqueeze_default_372 = None
        unsqueeze_default_374 = torch.ops.aten.unsqueeze.default(primals_368, -1)
        primals_368 = None
        unsqueeze_default_375 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_374, -1
        )
        unsqueeze_default_374 = None
        mul_tensor_657 = torch.ops.aten.mul.Tensor(
            mul_tensor_651, unsqueeze_default_373
        )
        mul_tensor_651 = unsqueeze_default_373 = None
        add_tensor_412 = torch.ops.aten.add.Tensor(
            mul_tensor_657, unsqueeze_default_375
        )
        mul_tensor_657 = unsqueeze_default_375 = None
        add_tensor_413 = torch.ops.aten.add.Tensor(add_tensor_404, add_tensor_412)
        add_tensor_404 = add_tensor_412 = None
        avg_pool2d_default_23 = torch.ops.aten.avg_pool2d.default(
            add_tensor_379, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_414 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_23, add_tensor_375
        )
        avg_pool2d_default_23 = None
        avg_pool2d_default_24 = torch.ops.aten.avg_pool2d.default(
            add_tensor_375, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_415 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_24, avg_pool2d_default_24
        )
        avg_pool2d_default_24 = None
        convolution_default_174 = torch.ops.aten.convolution.default(
            relu_default_85,
            primals_369,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        relu_default_85 = None
        convolution_default_175 = torch.ops.aten.convolution.default(
            convolution_default_174,
            primals_370,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_188 = torch.ops.aten.var.correction(
            convolution_default_175, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_94 = torch.ops.aten.mean.dim(convolution_default_175, [0, 2, 3], True)
        add_tensor_416 = torch.ops.aten.add.Tensor(var_correction_188, 0.001)
        var_correction_188 = None
        sqrt_default_94 = torch.ops.aten.sqrt.default(add_tensor_416)
        add_tensor_416 = None
        reciprocal_default_94 = torch.ops.aten.reciprocal.default(sqrt_default_94)
        sqrt_default_94 = None
        sub_tensor_94 = torch.ops.aten.sub.Tensor(convolution_default_175, mean_dim_94)
        mul_tensor_658 = torch.ops.aten.mul.Tensor(sub_tensor_94, reciprocal_default_94)
        sub_tensor_94 = None
        squeeze_dim_564 = torch.ops.aten.squeeze.dim(mean_dim_94, 3)
        mean_dim_94 = None
        squeeze_dim_565 = torch.ops.aten.squeeze.dim(squeeze_dim_564, 2)
        squeeze_dim_564 = None
        squeeze_dim_566 = torch.ops.aten.squeeze.dim(squeeze_dim_565, 0)
        squeeze_dim_565 = None
        squeeze_dim_567 = torch.ops.aten.squeeze.dim(reciprocal_default_94, 3)
        reciprocal_default_94 = None
        squeeze_dim_568 = torch.ops.aten.squeeze.dim(squeeze_dim_567, 2)
        squeeze_dim_567 = None
        squeeze_dim_569 = torch.ops.aten.squeeze.dim(squeeze_dim_568, 0)
        squeeze_dim_568 = None
        unsqueeze_default_376 = torch.ops.aten.unsqueeze.default(primals_371, -1)
        unsqueeze_default_377 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_376, -1
        )
        unsqueeze_default_376 = None
        unsqueeze_default_378 = torch.ops.aten.unsqueeze.default(primals_372, -1)
        primals_372 = None
        unsqueeze_default_379 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_378, -1
        )
        unsqueeze_default_378 = None
        mul_tensor_664 = torch.ops.aten.mul.Tensor(
            mul_tensor_658, unsqueeze_default_377
        )
        mul_tensor_658 = unsqueeze_default_377 = None
        add_tensor_419 = torch.ops.aten.add.Tensor(
            mul_tensor_664, unsqueeze_default_379
        )
        mul_tensor_664 = unsqueeze_default_379 = None
        relu_default_94 = torch.ops.aten.relu.default(add_tensor_419)
        add_tensor_419 = None
        convolution_default_176 = torch.ops.aten.convolution.default(
            relu_default_94,
            primals_373,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            168,
        )
        convolution_default_177 = torch.ops.aten.convolution.default(
            convolution_default_176,
            primals_374,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_190 = torch.ops.aten.var.correction(
            convolution_default_177, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_95 = torch.ops.aten.mean.dim(convolution_default_177, [0, 2, 3], True)
        add_tensor_420 = torch.ops.aten.add.Tensor(var_correction_190, 0.001)
        var_correction_190 = None
        sqrt_default_95 = torch.ops.aten.sqrt.default(add_tensor_420)
        add_tensor_420 = None
        reciprocal_default_95 = torch.ops.aten.reciprocal.default(sqrt_default_95)
        sqrt_default_95 = None
        sub_tensor_95 = torch.ops.aten.sub.Tensor(convolution_default_177, mean_dim_95)
        mul_tensor_665 = torch.ops.aten.mul.Tensor(sub_tensor_95, reciprocal_default_95)
        sub_tensor_95 = None
        squeeze_dim_570 = torch.ops.aten.squeeze.dim(mean_dim_95, 3)
        mean_dim_95 = None
        squeeze_dim_571 = torch.ops.aten.squeeze.dim(squeeze_dim_570, 2)
        squeeze_dim_570 = None
        squeeze_dim_572 = torch.ops.aten.squeeze.dim(squeeze_dim_571, 0)
        squeeze_dim_571 = None
        squeeze_dim_573 = torch.ops.aten.squeeze.dim(reciprocal_default_95, 3)
        reciprocal_default_95 = None
        squeeze_dim_574 = torch.ops.aten.squeeze.dim(squeeze_dim_573, 2)
        squeeze_dim_573 = None
        squeeze_dim_575 = torch.ops.aten.squeeze.dim(squeeze_dim_574, 0)
        squeeze_dim_574 = None
        unsqueeze_default_380 = torch.ops.aten.unsqueeze.default(primals_375, -1)
        unsqueeze_default_381 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_380, -1
        )
        unsqueeze_default_380 = None
        unsqueeze_default_382 = torch.ops.aten.unsqueeze.default(primals_376, -1)
        primals_376 = None
        unsqueeze_default_383 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_382, -1
        )
        unsqueeze_default_382 = None
        mul_tensor_671 = torch.ops.aten.mul.Tensor(
            mul_tensor_665, unsqueeze_default_381
        )
        mul_tensor_665 = unsqueeze_default_381 = None
        add_tensor_423 = torch.ops.aten.add.Tensor(
            mul_tensor_671, unsqueeze_default_383
        )
        mul_tensor_671 = unsqueeze_default_383 = None
        add_tensor_424 = torch.ops.aten.add.Tensor(add_tensor_423, add_tensor_379)
        add_tensor_423 = add_tensor_379 = None
        cat_default_9 = torch.ops.aten.cat.default(
            [
                add_tensor_375,
                add_tensor_396,
                add_tensor_413,
                add_tensor_414,
                add_tensor_415,
                add_tensor_424,
            ],
            1,
        )
        add_tensor_375 = (
            add_tensor_396
        ) = add_tensor_413 = add_tensor_414 = add_tensor_415 = add_tensor_424 = None
        convolution_default_178 = torch.ops.aten.convolution.default(
            relu_default_84, primals_377, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_192 = torch.ops.aten.var.correction(
            convolution_default_178, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_96 = torch.ops.aten.mean.dim(convolution_default_178, [0, 2, 3], True)
        add_tensor_425 = torch.ops.aten.add.Tensor(var_correction_192, 0.001)
        var_correction_192 = None
        sqrt_default_96 = torch.ops.aten.sqrt.default(add_tensor_425)
        add_tensor_425 = None
        reciprocal_default_96 = torch.ops.aten.reciprocal.default(sqrt_default_96)
        sqrt_default_96 = None
        sub_tensor_96 = torch.ops.aten.sub.Tensor(convolution_default_178, mean_dim_96)
        mul_tensor_672 = torch.ops.aten.mul.Tensor(sub_tensor_96, reciprocal_default_96)
        sub_tensor_96 = None
        squeeze_dim_576 = torch.ops.aten.squeeze.dim(mean_dim_96, 3)
        mean_dim_96 = None
        squeeze_dim_577 = torch.ops.aten.squeeze.dim(squeeze_dim_576, 2)
        squeeze_dim_576 = None
        squeeze_dim_578 = torch.ops.aten.squeeze.dim(squeeze_dim_577, 0)
        squeeze_dim_577 = None
        squeeze_dim_579 = torch.ops.aten.squeeze.dim(reciprocal_default_96, 3)
        reciprocal_default_96 = None
        squeeze_dim_580 = torch.ops.aten.squeeze.dim(squeeze_dim_579, 2)
        squeeze_dim_579 = None
        squeeze_dim_581 = torch.ops.aten.squeeze.dim(squeeze_dim_580, 0)
        squeeze_dim_580 = None
        unsqueeze_default_384 = torch.ops.aten.unsqueeze.default(primals_378, -1)
        unsqueeze_default_385 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_384, -1
        )
        unsqueeze_default_384 = None
        unsqueeze_default_386 = torch.ops.aten.unsqueeze.default(primals_379, -1)
        primals_379 = None
        unsqueeze_default_387 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_386, -1
        )
        unsqueeze_default_386 = None
        mul_tensor_678 = torch.ops.aten.mul.Tensor(
            mul_tensor_672, unsqueeze_default_385
        )
        mul_tensor_672 = unsqueeze_default_385 = None
        add_tensor_428 = torch.ops.aten.add.Tensor(
            mul_tensor_678, unsqueeze_default_387
        )
        mul_tensor_678 = unsqueeze_default_387 = None
        relu_default_96 = torch.ops.aten.relu.default(cat_default_9)
        cat_default_9 = None
        convolution_default_179 = torch.ops.aten.convolution.default(
            relu_default_96, primals_380, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )
        var_correction_194 = torch.ops.aten.var.correction(
            convolution_default_179, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_97 = torch.ops.aten.mean.dim(convolution_default_179, [0, 2, 3], True)
        add_tensor_429 = torch.ops.aten.add.Tensor(var_correction_194, 0.001)
        var_correction_194 = None
        sqrt_default_97 = torch.ops.aten.sqrt.default(add_tensor_429)
        add_tensor_429 = None
        reciprocal_default_97 = torch.ops.aten.reciprocal.default(sqrt_default_97)
        sqrt_default_97 = None
        sub_tensor_97 = torch.ops.aten.sub.Tensor(convolution_default_179, mean_dim_97)
        mul_tensor_679 = torch.ops.aten.mul.Tensor(sub_tensor_97, reciprocal_default_97)
        sub_tensor_97 = None
        squeeze_dim_582 = torch.ops.aten.squeeze.dim(mean_dim_97, 3)
        mean_dim_97 = None
        squeeze_dim_583 = torch.ops.aten.squeeze.dim(squeeze_dim_582, 2)
        squeeze_dim_582 = None
        squeeze_dim_584 = torch.ops.aten.squeeze.dim(squeeze_dim_583, 0)
        squeeze_dim_583 = None
        squeeze_dim_585 = torch.ops.aten.squeeze.dim(reciprocal_default_97, 3)
        reciprocal_default_97 = None
        squeeze_dim_586 = torch.ops.aten.squeeze.dim(squeeze_dim_585, 2)
        squeeze_dim_585 = None
        squeeze_dim_587 = torch.ops.aten.squeeze.dim(squeeze_dim_586, 0)
        squeeze_dim_586 = None
        unsqueeze_default_388 = torch.ops.aten.unsqueeze.default(primals_381, -1)
        unsqueeze_default_389 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_388, -1
        )
        unsqueeze_default_388 = None
        unsqueeze_default_390 = torch.ops.aten.unsqueeze.default(primals_382, -1)
        primals_382 = None
        unsqueeze_default_391 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_390, -1
        )
        unsqueeze_default_390 = None
        mul_tensor_685 = torch.ops.aten.mul.Tensor(
            mul_tensor_679, unsqueeze_default_389
        )
        mul_tensor_679 = unsqueeze_default_389 = None
        add_tensor_432 = torch.ops.aten.add.Tensor(
            mul_tensor_685, unsqueeze_default_391
        )
        mul_tensor_685 = unsqueeze_default_391 = None
        relu_default_97 = torch.ops.aten.relu.default(add_tensor_432)
        constant_pad_nd_default_16 = torch.ops.aten.constant_pad_nd.default(
            relu_default_97, [1, 2, 1, 2], 0.0
        )
        convolution_default_180 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_16,
            primals_9,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_181 = torch.ops.aten.convolution.default(
            convolution_default_180,
            primals_383,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_196 = torch.ops.aten.var.correction(
            convolution_default_181, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_98 = torch.ops.aten.mean.dim(convolution_default_181, [0, 2, 3], True)
        add_tensor_433 = torch.ops.aten.add.Tensor(var_correction_196, 0.001)
        var_correction_196 = None
        sqrt_default_98 = torch.ops.aten.sqrt.default(add_tensor_433)
        add_tensor_433 = None
        reciprocal_default_98 = torch.ops.aten.reciprocal.default(sqrt_default_98)
        sqrt_default_98 = None
        sub_tensor_98 = torch.ops.aten.sub.Tensor(convolution_default_181, mean_dim_98)
        mul_tensor_686 = torch.ops.aten.mul.Tensor(sub_tensor_98, reciprocal_default_98)
        sub_tensor_98 = None
        squeeze_dim_588 = torch.ops.aten.squeeze.dim(mean_dim_98, 3)
        mean_dim_98 = None
        squeeze_dim_589 = torch.ops.aten.squeeze.dim(squeeze_dim_588, 2)
        squeeze_dim_588 = None
        squeeze_dim_590 = torch.ops.aten.squeeze.dim(squeeze_dim_589, 0)
        squeeze_dim_589 = None
        squeeze_dim_591 = torch.ops.aten.squeeze.dim(reciprocal_default_98, 3)
        reciprocal_default_98 = None
        squeeze_dim_592 = torch.ops.aten.squeeze.dim(squeeze_dim_591, 2)
        squeeze_dim_591 = None
        squeeze_dim_593 = torch.ops.aten.squeeze.dim(squeeze_dim_592, 0)
        squeeze_dim_592 = None
        unsqueeze_default_392 = torch.ops.aten.unsqueeze.default(primals_384, -1)
        unsqueeze_default_393 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_392, -1
        )
        unsqueeze_default_392 = None
        unsqueeze_default_394 = torch.ops.aten.unsqueeze.default(primals_385, -1)
        primals_385 = None
        unsqueeze_default_395 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_394, -1
        )
        unsqueeze_default_394 = None
        mul_tensor_692 = torch.ops.aten.mul.Tensor(
            mul_tensor_686, unsqueeze_default_393
        )
        mul_tensor_686 = unsqueeze_default_393 = None
        add_tensor_436 = torch.ops.aten.add.Tensor(
            mul_tensor_692, unsqueeze_default_395
        )
        mul_tensor_692 = unsqueeze_default_395 = None
        relu_default_98 = torch.ops.aten.relu.default(add_tensor_436)
        add_tensor_436 = None
        convolution_default_182 = torch.ops.aten.convolution.default(
            relu_default_98,
            primals_386,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_183 = torch.ops.aten.convolution.default(
            convolution_default_182,
            primals_387,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_198 = torch.ops.aten.var.correction(
            convolution_default_183, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_99 = torch.ops.aten.mean.dim(convolution_default_183, [0, 2, 3], True)
        add_tensor_437 = torch.ops.aten.add.Tensor(var_correction_198, 0.001)
        var_correction_198 = None
        sqrt_default_99 = torch.ops.aten.sqrt.default(add_tensor_437)
        add_tensor_437 = None
        reciprocal_default_99 = torch.ops.aten.reciprocal.default(sqrt_default_99)
        sqrt_default_99 = None
        sub_tensor_99 = torch.ops.aten.sub.Tensor(convolution_default_183, mean_dim_99)
        mul_tensor_693 = torch.ops.aten.mul.Tensor(sub_tensor_99, reciprocal_default_99)
        sub_tensor_99 = None
        squeeze_dim_594 = torch.ops.aten.squeeze.dim(mean_dim_99, 3)
        mean_dim_99 = None
        squeeze_dim_595 = torch.ops.aten.squeeze.dim(squeeze_dim_594, 2)
        squeeze_dim_594 = None
        squeeze_dim_596 = torch.ops.aten.squeeze.dim(squeeze_dim_595, 0)
        squeeze_dim_595 = None
        squeeze_dim_597 = torch.ops.aten.squeeze.dim(reciprocal_default_99, 3)
        reciprocal_default_99 = None
        squeeze_dim_598 = torch.ops.aten.squeeze.dim(squeeze_dim_597, 2)
        squeeze_dim_597 = None
        squeeze_dim_599 = torch.ops.aten.squeeze.dim(squeeze_dim_598, 0)
        squeeze_dim_598 = None
        unsqueeze_default_396 = torch.ops.aten.unsqueeze.default(primals_388, -1)
        unsqueeze_default_397 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_396, -1
        )
        unsqueeze_default_396 = None
        unsqueeze_default_398 = torch.ops.aten.unsqueeze.default(primals_389, -1)
        primals_389 = None
        unsqueeze_default_399 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_398, -1
        )
        unsqueeze_default_398 = None
        mul_tensor_699 = torch.ops.aten.mul.Tensor(
            mul_tensor_693, unsqueeze_default_397
        )
        mul_tensor_693 = unsqueeze_default_397 = None
        add_tensor_440 = torch.ops.aten.add.Tensor(
            mul_tensor_699, unsqueeze_default_399
        )
        mul_tensor_699 = unsqueeze_default_399 = None
        relu_default_99 = torch.ops.aten.relu.default(add_tensor_428)
        add_tensor_428 = None
        constant_pad_nd_default_17 = torch.ops.aten.constant_pad_nd.default(
            relu_default_99, [2, 3, 2, 3], 0.0
        )
        convolution_default_184 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_17,
            primals_10,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_185 = torch.ops.aten.convolution.default(
            convolution_default_184,
            primals_390,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_200 = torch.ops.aten.var.correction(
            convolution_default_185, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_100 = torch.ops.aten.mean.dim(convolution_default_185, [0, 2, 3], True)
        add_tensor_441 = torch.ops.aten.add.Tensor(var_correction_200, 0.001)
        var_correction_200 = None
        sqrt_default_100 = torch.ops.aten.sqrt.default(add_tensor_441)
        add_tensor_441 = None
        reciprocal_default_100 = torch.ops.aten.reciprocal.default(sqrt_default_100)
        sqrt_default_100 = None
        sub_tensor_100 = torch.ops.aten.sub.Tensor(
            convolution_default_185, mean_dim_100
        )
        mul_tensor_700 = torch.ops.aten.mul.Tensor(
            sub_tensor_100, reciprocal_default_100
        )
        sub_tensor_100 = None
        squeeze_dim_600 = torch.ops.aten.squeeze.dim(mean_dim_100, 3)
        mean_dim_100 = None
        squeeze_dim_601 = torch.ops.aten.squeeze.dim(squeeze_dim_600, 2)
        squeeze_dim_600 = None
        squeeze_dim_602 = torch.ops.aten.squeeze.dim(squeeze_dim_601, 0)
        squeeze_dim_601 = None
        squeeze_dim_603 = torch.ops.aten.squeeze.dim(reciprocal_default_100, 3)
        reciprocal_default_100 = None
        squeeze_dim_604 = torch.ops.aten.squeeze.dim(squeeze_dim_603, 2)
        squeeze_dim_603 = None
        squeeze_dim_605 = torch.ops.aten.squeeze.dim(squeeze_dim_604, 0)
        squeeze_dim_604 = None
        unsqueeze_default_400 = torch.ops.aten.unsqueeze.default(primals_391, -1)
        unsqueeze_default_401 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_400, -1
        )
        unsqueeze_default_400 = None
        unsqueeze_default_402 = torch.ops.aten.unsqueeze.default(primals_392, -1)
        primals_392 = None
        unsqueeze_default_403 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_402, -1
        )
        unsqueeze_default_402 = None
        mul_tensor_706 = torch.ops.aten.mul.Tensor(
            mul_tensor_700, unsqueeze_default_401
        )
        mul_tensor_700 = unsqueeze_default_401 = None
        add_tensor_444 = torch.ops.aten.add.Tensor(
            mul_tensor_706, unsqueeze_default_403
        )
        mul_tensor_706 = unsqueeze_default_403 = None
        relu_default_100 = torch.ops.aten.relu.default(add_tensor_444)
        add_tensor_444 = None
        convolution_default_186 = torch.ops.aten.convolution.default(
            relu_default_100,
            primals_393,
            None,
            [1, 1],
            [3, 3],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_187 = torch.ops.aten.convolution.default(
            convolution_default_186,
            primals_394,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_202 = torch.ops.aten.var.correction(
            convolution_default_187, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_101 = torch.ops.aten.mean.dim(convolution_default_187, [0, 2, 3], True)
        add_tensor_445 = torch.ops.aten.add.Tensor(var_correction_202, 0.001)
        var_correction_202 = None
        sqrt_default_101 = torch.ops.aten.sqrt.default(add_tensor_445)
        add_tensor_445 = None
        reciprocal_default_101 = torch.ops.aten.reciprocal.default(sqrt_default_101)
        sqrt_default_101 = None
        sub_tensor_101 = torch.ops.aten.sub.Tensor(
            convolution_default_187, mean_dim_101
        )
        mul_tensor_707 = torch.ops.aten.mul.Tensor(
            sub_tensor_101, reciprocal_default_101
        )
        sub_tensor_101 = None
        squeeze_dim_606 = torch.ops.aten.squeeze.dim(mean_dim_101, 3)
        mean_dim_101 = None
        squeeze_dim_607 = torch.ops.aten.squeeze.dim(squeeze_dim_606, 2)
        squeeze_dim_606 = None
        squeeze_dim_608 = torch.ops.aten.squeeze.dim(squeeze_dim_607, 0)
        squeeze_dim_607 = None
        squeeze_dim_609 = torch.ops.aten.squeeze.dim(reciprocal_default_101, 3)
        reciprocal_default_101 = None
        squeeze_dim_610 = torch.ops.aten.squeeze.dim(squeeze_dim_609, 2)
        squeeze_dim_609 = None
        squeeze_dim_611 = torch.ops.aten.squeeze.dim(squeeze_dim_610, 0)
        squeeze_dim_610 = None
        unsqueeze_default_404 = torch.ops.aten.unsqueeze.default(primals_395, -1)
        unsqueeze_default_405 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_404, -1
        )
        unsqueeze_default_404 = None
        unsqueeze_default_406 = torch.ops.aten.unsqueeze.default(primals_396, -1)
        primals_396 = None
        unsqueeze_default_407 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_406, -1
        )
        unsqueeze_default_406 = None
        mul_tensor_713 = torch.ops.aten.mul.Tensor(
            mul_tensor_707, unsqueeze_default_405
        )
        mul_tensor_707 = unsqueeze_default_405 = None
        add_tensor_448 = torch.ops.aten.add.Tensor(
            mul_tensor_713, unsqueeze_default_407
        )
        mul_tensor_713 = unsqueeze_default_407 = None
        add_tensor_449 = torch.ops.aten.add.Tensor(add_tensor_440, add_tensor_448)
        add_tensor_440 = add_tensor_448 = None
        constant_pad_nd_default_18 = torch.ops.aten.constant_pad_nd.default(
            add_tensor_432, [0, 1, 0, 1], -inf
        )
        max_pool2d_with_indices_default_4 = (
            torch.ops.aten.max_pool2d_with_indices.default(
                constant_pad_nd_default_18, [3, 3], [2, 2]
            )
        )
        getitem_8 = max_pool2d_with_indices_default_4[0]
        getitem_9 = max_pool2d_with_indices_default_4[1]
        max_pool2d_with_indices_default_4 = None
        convolution_default_188 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_17,
            primals_11,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_189 = torch.ops.aten.convolution.default(
            convolution_default_188,
            primals_397,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_204 = torch.ops.aten.var.correction(
            convolution_default_189, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_102 = torch.ops.aten.mean.dim(convolution_default_189, [0, 2, 3], True)
        add_tensor_450 = torch.ops.aten.add.Tensor(var_correction_204, 0.001)
        var_correction_204 = None
        sqrt_default_102 = torch.ops.aten.sqrt.default(add_tensor_450)
        add_tensor_450 = None
        reciprocal_default_102 = torch.ops.aten.reciprocal.default(sqrt_default_102)
        sqrt_default_102 = None
        sub_tensor_102 = torch.ops.aten.sub.Tensor(
            convolution_default_189, mean_dim_102
        )
        mul_tensor_714 = torch.ops.aten.mul.Tensor(
            sub_tensor_102, reciprocal_default_102
        )
        sub_tensor_102 = None
        squeeze_dim_612 = torch.ops.aten.squeeze.dim(mean_dim_102, 3)
        mean_dim_102 = None
        squeeze_dim_613 = torch.ops.aten.squeeze.dim(squeeze_dim_612, 2)
        squeeze_dim_612 = None
        squeeze_dim_614 = torch.ops.aten.squeeze.dim(squeeze_dim_613, 0)
        squeeze_dim_613 = None
        squeeze_dim_615 = torch.ops.aten.squeeze.dim(reciprocal_default_102, 3)
        reciprocal_default_102 = None
        squeeze_dim_616 = torch.ops.aten.squeeze.dim(squeeze_dim_615, 2)
        squeeze_dim_615 = None
        squeeze_dim_617 = torch.ops.aten.squeeze.dim(squeeze_dim_616, 0)
        squeeze_dim_616 = None
        unsqueeze_default_408 = torch.ops.aten.unsqueeze.default(primals_398, -1)
        unsqueeze_default_409 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_408, -1
        )
        unsqueeze_default_408 = None
        unsqueeze_default_410 = torch.ops.aten.unsqueeze.default(primals_399, -1)
        primals_399 = None
        unsqueeze_default_411 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_410, -1
        )
        unsqueeze_default_410 = None
        mul_tensor_720 = torch.ops.aten.mul.Tensor(
            mul_tensor_714, unsqueeze_default_409
        )
        mul_tensor_714 = unsqueeze_default_409 = None
        add_tensor_453 = torch.ops.aten.add.Tensor(
            mul_tensor_720, unsqueeze_default_411
        )
        mul_tensor_720 = unsqueeze_default_411 = None
        relu_default_102 = torch.ops.aten.relu.default(add_tensor_453)
        add_tensor_453 = None
        convolution_default_190 = torch.ops.aten.convolution.default(
            relu_default_102,
            primals_400,
            None,
            [1, 1],
            [3, 3],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_191 = torch.ops.aten.convolution.default(
            convolution_default_190,
            primals_401,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_206 = torch.ops.aten.var.correction(
            convolution_default_191, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_103 = torch.ops.aten.mean.dim(convolution_default_191, [0, 2, 3], True)
        add_tensor_454 = torch.ops.aten.add.Tensor(var_correction_206, 0.001)
        var_correction_206 = None
        sqrt_default_103 = torch.ops.aten.sqrt.default(add_tensor_454)
        add_tensor_454 = None
        reciprocal_default_103 = torch.ops.aten.reciprocal.default(sqrt_default_103)
        sqrt_default_103 = None
        sub_tensor_103 = torch.ops.aten.sub.Tensor(
            convolution_default_191, mean_dim_103
        )
        mul_tensor_721 = torch.ops.aten.mul.Tensor(
            sub_tensor_103, reciprocal_default_103
        )
        sub_tensor_103 = None
        squeeze_dim_618 = torch.ops.aten.squeeze.dim(mean_dim_103, 3)
        mean_dim_103 = None
        squeeze_dim_619 = torch.ops.aten.squeeze.dim(squeeze_dim_618, 2)
        squeeze_dim_618 = None
        squeeze_dim_620 = torch.ops.aten.squeeze.dim(squeeze_dim_619, 0)
        squeeze_dim_619 = None
        squeeze_dim_621 = torch.ops.aten.squeeze.dim(reciprocal_default_103, 3)
        reciprocal_default_103 = None
        squeeze_dim_622 = torch.ops.aten.squeeze.dim(squeeze_dim_621, 2)
        squeeze_dim_621 = None
        squeeze_dim_623 = torch.ops.aten.squeeze.dim(squeeze_dim_622, 0)
        squeeze_dim_622 = None
        unsqueeze_default_412 = torch.ops.aten.unsqueeze.default(primals_402, -1)
        unsqueeze_default_413 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_412, -1
        )
        unsqueeze_default_412 = None
        unsqueeze_default_414 = torch.ops.aten.unsqueeze.default(primals_403, -1)
        primals_403 = None
        unsqueeze_default_415 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_414, -1
        )
        unsqueeze_default_414 = None
        mul_tensor_727 = torch.ops.aten.mul.Tensor(
            mul_tensor_721, unsqueeze_default_413
        )
        mul_tensor_721 = unsqueeze_default_413 = None
        add_tensor_457 = torch.ops.aten.add.Tensor(
            mul_tensor_727, unsqueeze_default_415
        )
        mul_tensor_727 = unsqueeze_default_415 = None
        add_tensor_458 = torch.ops.aten.add.Tensor(getitem_8, add_tensor_457)
        add_tensor_457 = None
        constant_pad_nd_default_20 = torch.ops.aten.constant_pad_nd.default(
            add_tensor_432, [0, 1, 0, 1], 0.0
        )
        add_tensor_432 = None
        avg_pool2d_default_26 = torch.ops.aten.avg_pool2d.default(
            constant_pad_nd_default_20, [3, 3], [2, 2], [0, 0], False, False
        )
        constant_pad_nd_default_21 = torch.ops.aten.constant_pad_nd.default(
            relu_default_99, [1, 2, 1, 2], 0.0
        )
        convolution_default_192 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_21,
            primals_12,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_193 = torch.ops.aten.convolution.default(
            convolution_default_192,
            primals_404,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_208 = torch.ops.aten.var.correction(
            convolution_default_193, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_104 = torch.ops.aten.mean.dim(convolution_default_193, [0, 2, 3], True)
        add_tensor_459 = torch.ops.aten.add.Tensor(var_correction_208, 0.001)
        var_correction_208 = None
        sqrt_default_104 = torch.ops.aten.sqrt.default(add_tensor_459)
        add_tensor_459 = None
        reciprocal_default_104 = torch.ops.aten.reciprocal.default(sqrt_default_104)
        sqrt_default_104 = None
        sub_tensor_104 = torch.ops.aten.sub.Tensor(
            convolution_default_193, mean_dim_104
        )
        mul_tensor_728 = torch.ops.aten.mul.Tensor(
            sub_tensor_104, reciprocal_default_104
        )
        sub_tensor_104 = None
        squeeze_dim_624 = torch.ops.aten.squeeze.dim(mean_dim_104, 3)
        mean_dim_104 = None
        squeeze_dim_625 = torch.ops.aten.squeeze.dim(squeeze_dim_624, 2)
        squeeze_dim_624 = None
        squeeze_dim_626 = torch.ops.aten.squeeze.dim(squeeze_dim_625, 0)
        squeeze_dim_625 = None
        squeeze_dim_627 = torch.ops.aten.squeeze.dim(reciprocal_default_104, 3)
        reciprocal_default_104 = None
        squeeze_dim_628 = torch.ops.aten.squeeze.dim(squeeze_dim_627, 2)
        squeeze_dim_627 = None
        squeeze_dim_629 = torch.ops.aten.squeeze.dim(squeeze_dim_628, 0)
        squeeze_dim_628 = None
        unsqueeze_default_416 = torch.ops.aten.unsqueeze.default(primals_405, -1)
        unsqueeze_default_417 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_416, -1
        )
        unsqueeze_default_416 = None
        unsqueeze_default_418 = torch.ops.aten.unsqueeze.default(primals_406, -1)
        primals_406 = None
        unsqueeze_default_419 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_418, -1
        )
        unsqueeze_default_418 = None
        mul_tensor_734 = torch.ops.aten.mul.Tensor(
            mul_tensor_728, unsqueeze_default_417
        )
        mul_tensor_728 = unsqueeze_default_417 = None
        add_tensor_462 = torch.ops.aten.add.Tensor(
            mul_tensor_734, unsqueeze_default_419
        )
        mul_tensor_734 = unsqueeze_default_419 = None
        relu_default_104 = torch.ops.aten.relu.default(add_tensor_462)
        add_tensor_462 = None
        convolution_default_194 = torch.ops.aten.convolution.default(
            relu_default_104,
            primals_407,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_195 = torch.ops.aten.convolution.default(
            convolution_default_194,
            primals_408,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_210 = torch.ops.aten.var.correction(
            convolution_default_195, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_105 = torch.ops.aten.mean.dim(convolution_default_195, [0, 2, 3], True)
        add_tensor_463 = torch.ops.aten.add.Tensor(var_correction_210, 0.001)
        var_correction_210 = None
        sqrt_default_105 = torch.ops.aten.sqrt.default(add_tensor_463)
        add_tensor_463 = None
        reciprocal_default_105 = torch.ops.aten.reciprocal.default(sqrt_default_105)
        sqrt_default_105 = None
        sub_tensor_105 = torch.ops.aten.sub.Tensor(
            convolution_default_195, mean_dim_105
        )
        mul_tensor_735 = torch.ops.aten.mul.Tensor(
            sub_tensor_105, reciprocal_default_105
        )
        sub_tensor_105 = None
        squeeze_dim_630 = torch.ops.aten.squeeze.dim(mean_dim_105, 3)
        mean_dim_105 = None
        squeeze_dim_631 = torch.ops.aten.squeeze.dim(squeeze_dim_630, 2)
        squeeze_dim_630 = None
        squeeze_dim_632 = torch.ops.aten.squeeze.dim(squeeze_dim_631, 0)
        squeeze_dim_631 = None
        squeeze_dim_633 = torch.ops.aten.squeeze.dim(reciprocal_default_105, 3)
        reciprocal_default_105 = None
        squeeze_dim_634 = torch.ops.aten.squeeze.dim(squeeze_dim_633, 2)
        squeeze_dim_633 = None
        squeeze_dim_635 = torch.ops.aten.squeeze.dim(squeeze_dim_634, 0)
        squeeze_dim_634 = None
        unsqueeze_default_420 = torch.ops.aten.unsqueeze.default(primals_409, -1)
        unsqueeze_default_421 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_420, -1
        )
        unsqueeze_default_420 = None
        unsqueeze_default_422 = torch.ops.aten.unsqueeze.default(primals_410, -1)
        primals_410 = None
        unsqueeze_default_423 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_422, -1
        )
        unsqueeze_default_422 = None
        mul_tensor_741 = torch.ops.aten.mul.Tensor(
            mul_tensor_735, unsqueeze_default_421
        )
        mul_tensor_735 = unsqueeze_default_421 = None
        add_tensor_466 = torch.ops.aten.add.Tensor(
            mul_tensor_741, unsqueeze_default_423
        )
        mul_tensor_741 = unsqueeze_default_423 = None
        add_tensor_467 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_26, add_tensor_466
        )
        avg_pool2d_default_26 = add_tensor_466 = None
        avg_pool2d_default_27 = torch.ops.aten.avg_pool2d.default(
            add_tensor_449, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_468 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_27, add_tensor_458
        )
        avg_pool2d_default_27 = None
        relu_default_105 = torch.ops.aten.relu.default(add_tensor_449)
        convolution_default_196 = torch.ops.aten.convolution.default(
            relu_default_105,
            primals_411,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_105 = None
        convolution_default_197 = torch.ops.aten.convolution.default(
            convolution_default_196,
            primals_412,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_212 = torch.ops.aten.var.correction(
            convolution_default_197, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_106 = torch.ops.aten.mean.dim(convolution_default_197, [0, 2, 3], True)
        add_tensor_469 = torch.ops.aten.add.Tensor(var_correction_212, 0.001)
        var_correction_212 = None
        sqrt_default_106 = torch.ops.aten.sqrt.default(add_tensor_469)
        add_tensor_469 = None
        reciprocal_default_106 = torch.ops.aten.reciprocal.default(sqrt_default_106)
        sqrt_default_106 = None
        sub_tensor_106 = torch.ops.aten.sub.Tensor(
            convolution_default_197, mean_dim_106
        )
        mul_tensor_742 = torch.ops.aten.mul.Tensor(
            sub_tensor_106, reciprocal_default_106
        )
        sub_tensor_106 = None
        squeeze_dim_636 = torch.ops.aten.squeeze.dim(mean_dim_106, 3)
        mean_dim_106 = None
        squeeze_dim_637 = torch.ops.aten.squeeze.dim(squeeze_dim_636, 2)
        squeeze_dim_636 = None
        squeeze_dim_638 = torch.ops.aten.squeeze.dim(squeeze_dim_637, 0)
        squeeze_dim_637 = None
        squeeze_dim_639 = torch.ops.aten.squeeze.dim(reciprocal_default_106, 3)
        reciprocal_default_106 = None
        squeeze_dim_640 = torch.ops.aten.squeeze.dim(squeeze_dim_639, 2)
        squeeze_dim_639 = None
        squeeze_dim_641 = torch.ops.aten.squeeze.dim(squeeze_dim_640, 0)
        squeeze_dim_640 = None
        unsqueeze_default_424 = torch.ops.aten.unsqueeze.default(primals_413, -1)
        unsqueeze_default_425 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_424, -1
        )
        unsqueeze_default_424 = None
        unsqueeze_default_426 = torch.ops.aten.unsqueeze.default(primals_414, -1)
        primals_414 = None
        unsqueeze_default_427 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_426, -1
        )
        unsqueeze_default_426 = None
        mul_tensor_748 = torch.ops.aten.mul.Tensor(
            mul_tensor_742, unsqueeze_default_425
        )
        mul_tensor_742 = unsqueeze_default_425 = None
        add_tensor_472 = torch.ops.aten.add.Tensor(
            mul_tensor_748, unsqueeze_default_427
        )
        mul_tensor_748 = unsqueeze_default_427 = None
        relu_default_106 = torch.ops.aten.relu.default(add_tensor_472)
        add_tensor_472 = None
        convolution_default_198 = torch.ops.aten.convolution.default(
            relu_default_106,
            primals_415,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_199 = torch.ops.aten.convolution.default(
            convolution_default_198,
            primals_416,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_214 = torch.ops.aten.var.correction(
            convolution_default_199, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_107 = torch.ops.aten.mean.dim(convolution_default_199, [0, 2, 3], True)
        add_tensor_473 = torch.ops.aten.add.Tensor(var_correction_214, 0.001)
        var_correction_214 = None
        sqrt_default_107 = torch.ops.aten.sqrt.default(add_tensor_473)
        add_tensor_473 = None
        reciprocal_default_107 = torch.ops.aten.reciprocal.default(sqrt_default_107)
        sqrt_default_107 = None
        sub_tensor_107 = torch.ops.aten.sub.Tensor(
            convolution_default_199, mean_dim_107
        )
        mul_tensor_749 = torch.ops.aten.mul.Tensor(
            sub_tensor_107, reciprocal_default_107
        )
        sub_tensor_107 = None
        squeeze_dim_642 = torch.ops.aten.squeeze.dim(mean_dim_107, 3)
        mean_dim_107 = None
        squeeze_dim_643 = torch.ops.aten.squeeze.dim(squeeze_dim_642, 2)
        squeeze_dim_642 = None
        squeeze_dim_644 = torch.ops.aten.squeeze.dim(squeeze_dim_643, 0)
        squeeze_dim_643 = None
        squeeze_dim_645 = torch.ops.aten.squeeze.dim(reciprocal_default_107, 3)
        reciprocal_default_107 = None
        squeeze_dim_646 = torch.ops.aten.squeeze.dim(squeeze_dim_645, 2)
        squeeze_dim_645 = None
        squeeze_dim_647 = torch.ops.aten.squeeze.dim(squeeze_dim_646, 0)
        squeeze_dim_646 = None
        unsqueeze_default_428 = torch.ops.aten.unsqueeze.default(primals_417, -1)
        unsqueeze_default_429 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_428, -1
        )
        unsqueeze_default_428 = None
        unsqueeze_default_430 = torch.ops.aten.unsqueeze.default(primals_418, -1)
        primals_418 = None
        unsqueeze_default_431 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_430, -1
        )
        unsqueeze_default_430 = None
        mul_tensor_755 = torch.ops.aten.mul.Tensor(
            mul_tensor_749, unsqueeze_default_429
        )
        mul_tensor_749 = unsqueeze_default_429 = None
        add_tensor_476 = torch.ops.aten.add.Tensor(
            mul_tensor_755, unsqueeze_default_431
        )
        mul_tensor_755 = unsqueeze_default_431 = None
        add_tensor_477 = torch.ops.aten.add.Tensor(add_tensor_476, getitem_8)
        add_tensor_476 = getitem_8 = None
        cat_default_10 = torch.ops.aten.cat.default(
            [add_tensor_458, add_tensor_467, add_tensor_468, add_tensor_477], 1
        )
        add_tensor_458 = add_tensor_467 = add_tensor_468 = add_tensor_477 = None
        avg_pool2d_default_28 = torch.ops.aten.avg_pool2d.default(
            relu_default_84, [1, 1], [2, 2], [0, 0], False, False
        )
        convolution_default_200 = torch.ops.aten.convolution.default(
            avg_pool2d_default_28,
            primals_419,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        constant_pad_nd_default_23 = torch.ops.aten.constant_pad_nd.default(
            relu_default_84, [-1, 1, -1, 1], 0.0
        )
        avg_pool2d_default_29 = torch.ops.aten.avg_pool2d.default(
            constant_pad_nd_default_23, [1, 1], [2, 2], [0, 0], False, False
        )
        convolution_default_201 = torch.ops.aten.convolution.default(
            avg_pool2d_default_29,
            primals_420,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        cat_default_11 = torch.ops.aten.cat.default(
            [convolution_default_200, convolution_default_201], 1
        )
        convolution_default_200 = convolution_default_201 = None
        var_correction_216 = torch.ops.aten.var.correction(
            cat_default_11, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_108 = torch.ops.aten.mean.dim(cat_default_11, [0, 2, 3], True)
        add_tensor_478 = torch.ops.aten.add.Tensor(var_correction_216, 0.001)
        var_correction_216 = None
        sqrt_default_108 = torch.ops.aten.sqrt.default(add_tensor_478)
        add_tensor_478 = None
        reciprocal_default_108 = torch.ops.aten.reciprocal.default(sqrt_default_108)
        sqrt_default_108 = None
        sub_tensor_108 = torch.ops.aten.sub.Tensor(cat_default_11, mean_dim_108)
        mul_tensor_756 = torch.ops.aten.mul.Tensor(
            sub_tensor_108, reciprocal_default_108
        )
        sub_tensor_108 = None
        unsqueeze_default_432 = torch.ops.aten.unsqueeze.default(primals_421, -1)
        unsqueeze_default_433 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_432, -1
        )
        unsqueeze_default_432 = None
        unsqueeze_default_434 = torch.ops.aten.unsqueeze.default(primals_422, -1)
        unsqueeze_default_435 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_434, -1
        )
        unsqueeze_default_434 = None
        mul_tensor_762 = torch.ops.aten.mul.Tensor(
            mul_tensor_756, unsqueeze_default_433
        )
        mul_tensor_756 = unsqueeze_default_433 = None
        add_tensor_481 = torch.ops.aten.add.Tensor(
            mul_tensor_762, unsqueeze_default_435
        )
        mul_tensor_762 = unsqueeze_default_435 = None
        relu_default_108 = torch.ops.aten.relu.default(cat_default_10)
        cat_default_10 = None
        convolution_default_202 = torch.ops.aten.convolution.default(
            relu_default_108,
            primals_423,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_218 = torch.ops.aten.var.correction(
            convolution_default_202, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_109 = torch.ops.aten.mean.dim(convolution_default_202, [0, 2, 3], True)
        add_tensor_482 = torch.ops.aten.add.Tensor(var_correction_218, 0.001)
        var_correction_218 = None
        sqrt_default_109 = torch.ops.aten.sqrt.default(add_tensor_482)
        add_tensor_482 = None
        reciprocal_default_109 = torch.ops.aten.reciprocal.default(sqrt_default_109)
        sqrt_default_109 = None
        sub_tensor_109 = torch.ops.aten.sub.Tensor(
            convolution_default_202, mean_dim_109
        )
        mul_tensor_763 = torch.ops.aten.mul.Tensor(
            sub_tensor_109, reciprocal_default_109
        )
        sub_tensor_109 = None
        unsqueeze_default_436 = torch.ops.aten.unsqueeze.default(primals_424, -1)
        unsqueeze_default_437 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_436, -1
        )
        unsqueeze_default_436 = None
        unsqueeze_default_438 = torch.ops.aten.unsqueeze.default(primals_425, -1)
        unsqueeze_default_439 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_438, -1
        )
        unsqueeze_default_438 = None
        mul_tensor_769 = torch.ops.aten.mul.Tensor(
            mul_tensor_763, unsqueeze_default_437
        )
        mul_tensor_763 = unsqueeze_default_437 = None
        add_tensor_485 = torch.ops.aten.add.Tensor(
            mul_tensor_769, unsqueeze_default_439
        )
        mul_tensor_769 = unsqueeze_default_439 = None
        relu_default_109 = torch.ops.aten.relu.default(add_tensor_485)
        convolution_default_203 = torch.ops.aten.convolution.default(
            relu_default_109,
            primals_426,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_204 = torch.ops.aten.convolution.default(
            convolution_default_203,
            primals_427,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_220 = torch.ops.aten.var.correction(
            convolution_default_204, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_110 = torch.ops.aten.mean.dim(convolution_default_204, [0, 2, 3], True)
        add_tensor_486 = torch.ops.aten.add.Tensor(var_correction_220, 0.001)
        var_correction_220 = None
        sqrt_default_110 = torch.ops.aten.sqrt.default(add_tensor_486)
        add_tensor_486 = None
        reciprocal_default_110 = torch.ops.aten.reciprocal.default(sqrt_default_110)
        sqrt_default_110 = None
        sub_tensor_110 = torch.ops.aten.sub.Tensor(
            convolution_default_204, mean_dim_110
        )
        mul_tensor_770 = torch.ops.aten.mul.Tensor(
            sub_tensor_110, reciprocal_default_110
        )
        sub_tensor_110 = None
        squeeze_dim_660 = torch.ops.aten.squeeze.dim(mean_dim_110, 3)
        mean_dim_110 = None
        squeeze_dim_661 = torch.ops.aten.squeeze.dim(squeeze_dim_660, 2)
        squeeze_dim_660 = None
        squeeze_dim_662 = torch.ops.aten.squeeze.dim(squeeze_dim_661, 0)
        squeeze_dim_661 = None
        squeeze_dim_663 = torch.ops.aten.squeeze.dim(reciprocal_default_110, 3)
        reciprocal_default_110 = None
        squeeze_dim_664 = torch.ops.aten.squeeze.dim(squeeze_dim_663, 2)
        squeeze_dim_663 = None
        squeeze_dim_665 = torch.ops.aten.squeeze.dim(squeeze_dim_664, 0)
        squeeze_dim_664 = None
        unsqueeze_default_440 = torch.ops.aten.unsqueeze.default(primals_428, -1)
        unsqueeze_default_441 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_440, -1
        )
        unsqueeze_default_440 = None
        unsqueeze_default_442 = torch.ops.aten.unsqueeze.default(primals_429, -1)
        primals_429 = None
        unsqueeze_default_443 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_442, -1
        )
        unsqueeze_default_442 = None
        mul_tensor_776 = torch.ops.aten.mul.Tensor(
            mul_tensor_770, unsqueeze_default_441
        )
        mul_tensor_770 = unsqueeze_default_441 = None
        add_tensor_489 = torch.ops.aten.add.Tensor(
            mul_tensor_776, unsqueeze_default_443
        )
        mul_tensor_776 = unsqueeze_default_443 = None
        relu_default_110 = torch.ops.aten.relu.default(add_tensor_489)
        add_tensor_489 = None
        convolution_default_205 = torch.ops.aten.convolution.default(
            relu_default_110,
            primals_430,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_206 = torch.ops.aten.convolution.default(
            convolution_default_205,
            primals_431,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_222 = torch.ops.aten.var.correction(
            convolution_default_206, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_111 = torch.ops.aten.mean.dim(convolution_default_206, [0, 2, 3], True)
        add_tensor_490 = torch.ops.aten.add.Tensor(var_correction_222, 0.001)
        var_correction_222 = None
        sqrt_default_111 = torch.ops.aten.sqrt.default(add_tensor_490)
        add_tensor_490 = None
        reciprocal_default_111 = torch.ops.aten.reciprocal.default(sqrt_default_111)
        sqrt_default_111 = None
        sub_tensor_111 = torch.ops.aten.sub.Tensor(
            convolution_default_206, mean_dim_111
        )
        mul_tensor_777 = torch.ops.aten.mul.Tensor(
            sub_tensor_111, reciprocal_default_111
        )
        sub_tensor_111 = None
        squeeze_dim_666 = torch.ops.aten.squeeze.dim(mean_dim_111, 3)
        mean_dim_111 = None
        squeeze_dim_667 = torch.ops.aten.squeeze.dim(squeeze_dim_666, 2)
        squeeze_dim_666 = None
        squeeze_dim_668 = torch.ops.aten.squeeze.dim(squeeze_dim_667, 0)
        squeeze_dim_667 = None
        squeeze_dim_669 = torch.ops.aten.squeeze.dim(reciprocal_default_111, 3)
        reciprocal_default_111 = None
        squeeze_dim_670 = torch.ops.aten.squeeze.dim(squeeze_dim_669, 2)
        squeeze_dim_669 = None
        squeeze_dim_671 = torch.ops.aten.squeeze.dim(squeeze_dim_670, 0)
        squeeze_dim_670 = None
        unsqueeze_default_444 = torch.ops.aten.unsqueeze.default(primals_432, -1)
        unsqueeze_default_445 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_444, -1
        )
        unsqueeze_default_444 = None
        unsqueeze_default_446 = torch.ops.aten.unsqueeze.default(primals_433, -1)
        primals_433 = None
        unsqueeze_default_447 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_446, -1
        )
        unsqueeze_default_446 = None
        mul_tensor_783 = torch.ops.aten.mul.Tensor(
            mul_tensor_777, unsqueeze_default_445
        )
        mul_tensor_777 = unsqueeze_default_445 = None
        add_tensor_493 = torch.ops.aten.add.Tensor(
            mul_tensor_783, unsqueeze_default_447
        )
        mul_tensor_783 = unsqueeze_default_447 = None
        relu_default_111 = torch.ops.aten.relu.default(add_tensor_481)
        convolution_default_207 = torch.ops.aten.convolution.default(
            relu_default_111,
            primals_434,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_208 = torch.ops.aten.convolution.default(
            convolution_default_207,
            primals_435,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_224 = torch.ops.aten.var.correction(
            convolution_default_208, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_112 = torch.ops.aten.mean.dim(convolution_default_208, [0, 2, 3], True)
        add_tensor_494 = torch.ops.aten.add.Tensor(var_correction_224, 0.001)
        var_correction_224 = None
        sqrt_default_112 = torch.ops.aten.sqrt.default(add_tensor_494)
        add_tensor_494 = None
        reciprocal_default_112 = torch.ops.aten.reciprocal.default(sqrt_default_112)
        sqrt_default_112 = None
        sub_tensor_112 = torch.ops.aten.sub.Tensor(
            convolution_default_208, mean_dim_112
        )
        mul_tensor_784 = torch.ops.aten.mul.Tensor(
            sub_tensor_112, reciprocal_default_112
        )
        sub_tensor_112 = None
        squeeze_dim_672 = torch.ops.aten.squeeze.dim(mean_dim_112, 3)
        mean_dim_112 = None
        squeeze_dim_673 = torch.ops.aten.squeeze.dim(squeeze_dim_672, 2)
        squeeze_dim_672 = None
        squeeze_dim_674 = torch.ops.aten.squeeze.dim(squeeze_dim_673, 0)
        squeeze_dim_673 = None
        squeeze_dim_675 = torch.ops.aten.squeeze.dim(reciprocal_default_112, 3)
        reciprocal_default_112 = None
        squeeze_dim_676 = torch.ops.aten.squeeze.dim(squeeze_dim_675, 2)
        squeeze_dim_675 = None
        squeeze_dim_677 = torch.ops.aten.squeeze.dim(squeeze_dim_676, 0)
        squeeze_dim_676 = None
        unsqueeze_default_448 = torch.ops.aten.unsqueeze.default(primals_436, -1)
        unsqueeze_default_449 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_448, -1
        )
        unsqueeze_default_448 = None
        unsqueeze_default_450 = torch.ops.aten.unsqueeze.default(primals_437, -1)
        primals_437 = None
        unsqueeze_default_451 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_450, -1
        )
        unsqueeze_default_450 = None
        mul_tensor_790 = torch.ops.aten.mul.Tensor(
            mul_tensor_784, unsqueeze_default_449
        )
        mul_tensor_784 = unsqueeze_default_449 = None
        add_tensor_497 = torch.ops.aten.add.Tensor(
            mul_tensor_790, unsqueeze_default_451
        )
        mul_tensor_790 = unsqueeze_default_451 = None
        relu_default_112 = torch.ops.aten.relu.default(add_tensor_497)
        add_tensor_497 = None
        convolution_default_209 = torch.ops.aten.convolution.default(
            relu_default_112,
            primals_438,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_210 = torch.ops.aten.convolution.default(
            convolution_default_209,
            primals_439,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_226 = torch.ops.aten.var.correction(
            convolution_default_210, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_113 = torch.ops.aten.mean.dim(convolution_default_210, [0, 2, 3], True)
        add_tensor_498 = torch.ops.aten.add.Tensor(var_correction_226, 0.001)
        var_correction_226 = None
        sqrt_default_113 = torch.ops.aten.sqrt.default(add_tensor_498)
        add_tensor_498 = None
        reciprocal_default_113 = torch.ops.aten.reciprocal.default(sqrt_default_113)
        sqrt_default_113 = None
        sub_tensor_113 = torch.ops.aten.sub.Tensor(
            convolution_default_210, mean_dim_113
        )
        mul_tensor_791 = torch.ops.aten.mul.Tensor(
            sub_tensor_113, reciprocal_default_113
        )
        sub_tensor_113 = None
        squeeze_dim_678 = torch.ops.aten.squeeze.dim(mean_dim_113, 3)
        mean_dim_113 = None
        squeeze_dim_679 = torch.ops.aten.squeeze.dim(squeeze_dim_678, 2)
        squeeze_dim_678 = None
        squeeze_dim_680 = torch.ops.aten.squeeze.dim(squeeze_dim_679, 0)
        squeeze_dim_679 = None
        squeeze_dim_681 = torch.ops.aten.squeeze.dim(reciprocal_default_113, 3)
        reciprocal_default_113 = None
        squeeze_dim_682 = torch.ops.aten.squeeze.dim(squeeze_dim_681, 2)
        squeeze_dim_681 = None
        squeeze_dim_683 = torch.ops.aten.squeeze.dim(squeeze_dim_682, 0)
        squeeze_dim_682 = None
        unsqueeze_default_452 = torch.ops.aten.unsqueeze.default(primals_440, -1)
        unsqueeze_default_453 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_452, -1
        )
        unsqueeze_default_452 = None
        unsqueeze_default_454 = torch.ops.aten.unsqueeze.default(primals_441, -1)
        primals_441 = None
        unsqueeze_default_455 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_454, -1
        )
        unsqueeze_default_454 = None
        mul_tensor_797 = torch.ops.aten.mul.Tensor(
            mul_tensor_791, unsqueeze_default_453
        )
        mul_tensor_791 = unsqueeze_default_453 = None
        add_tensor_501 = torch.ops.aten.add.Tensor(
            mul_tensor_797, unsqueeze_default_455
        )
        mul_tensor_797 = unsqueeze_default_455 = None
        add_tensor_502 = torch.ops.aten.add.Tensor(add_tensor_493, add_tensor_501)
        add_tensor_493 = add_tensor_501 = None
        convolution_default_211 = torch.ops.aten.convolution.default(
            relu_default_111,
            primals_442,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_212 = torch.ops.aten.convolution.default(
            convolution_default_211,
            primals_443,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_228 = torch.ops.aten.var.correction(
            convolution_default_212, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_114 = torch.ops.aten.mean.dim(convolution_default_212, [0, 2, 3], True)
        add_tensor_503 = torch.ops.aten.add.Tensor(var_correction_228, 0.001)
        var_correction_228 = None
        sqrt_default_114 = torch.ops.aten.sqrt.default(add_tensor_503)
        add_tensor_503 = None
        reciprocal_default_114 = torch.ops.aten.reciprocal.default(sqrt_default_114)
        sqrt_default_114 = None
        sub_tensor_114 = torch.ops.aten.sub.Tensor(
            convolution_default_212, mean_dim_114
        )
        mul_tensor_798 = torch.ops.aten.mul.Tensor(
            sub_tensor_114, reciprocal_default_114
        )
        sub_tensor_114 = None
        squeeze_dim_684 = torch.ops.aten.squeeze.dim(mean_dim_114, 3)
        mean_dim_114 = None
        squeeze_dim_685 = torch.ops.aten.squeeze.dim(squeeze_dim_684, 2)
        squeeze_dim_684 = None
        squeeze_dim_686 = torch.ops.aten.squeeze.dim(squeeze_dim_685, 0)
        squeeze_dim_685 = None
        squeeze_dim_687 = torch.ops.aten.squeeze.dim(reciprocal_default_114, 3)
        reciprocal_default_114 = None
        squeeze_dim_688 = torch.ops.aten.squeeze.dim(squeeze_dim_687, 2)
        squeeze_dim_687 = None
        squeeze_dim_689 = torch.ops.aten.squeeze.dim(squeeze_dim_688, 0)
        squeeze_dim_688 = None
        unsqueeze_default_456 = torch.ops.aten.unsqueeze.default(primals_444, -1)
        unsqueeze_default_457 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_456, -1
        )
        unsqueeze_default_456 = None
        unsqueeze_default_458 = torch.ops.aten.unsqueeze.default(primals_445, -1)
        primals_445 = None
        unsqueeze_default_459 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_458, -1
        )
        unsqueeze_default_458 = None
        mul_tensor_804 = torch.ops.aten.mul.Tensor(
            mul_tensor_798, unsqueeze_default_457
        )
        mul_tensor_798 = unsqueeze_default_457 = None
        add_tensor_506 = torch.ops.aten.add.Tensor(
            mul_tensor_804, unsqueeze_default_459
        )
        mul_tensor_804 = unsqueeze_default_459 = None
        relu_default_114 = torch.ops.aten.relu.default(add_tensor_506)
        add_tensor_506 = None
        convolution_default_213 = torch.ops.aten.convolution.default(
            relu_default_114,
            primals_446,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_214 = torch.ops.aten.convolution.default(
            convolution_default_213,
            primals_447,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_230 = torch.ops.aten.var.correction(
            convolution_default_214, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_115 = torch.ops.aten.mean.dim(convolution_default_214, [0, 2, 3], True)
        add_tensor_507 = torch.ops.aten.add.Tensor(var_correction_230, 0.001)
        var_correction_230 = None
        sqrt_default_115 = torch.ops.aten.sqrt.default(add_tensor_507)
        add_tensor_507 = None
        reciprocal_default_115 = torch.ops.aten.reciprocal.default(sqrt_default_115)
        sqrt_default_115 = None
        sub_tensor_115 = torch.ops.aten.sub.Tensor(
            convolution_default_214, mean_dim_115
        )
        mul_tensor_805 = torch.ops.aten.mul.Tensor(
            sub_tensor_115, reciprocal_default_115
        )
        sub_tensor_115 = None
        squeeze_dim_690 = torch.ops.aten.squeeze.dim(mean_dim_115, 3)
        mean_dim_115 = None
        squeeze_dim_691 = torch.ops.aten.squeeze.dim(squeeze_dim_690, 2)
        squeeze_dim_690 = None
        squeeze_dim_692 = torch.ops.aten.squeeze.dim(squeeze_dim_691, 0)
        squeeze_dim_691 = None
        squeeze_dim_693 = torch.ops.aten.squeeze.dim(reciprocal_default_115, 3)
        reciprocal_default_115 = None
        squeeze_dim_694 = torch.ops.aten.squeeze.dim(squeeze_dim_693, 2)
        squeeze_dim_693 = None
        squeeze_dim_695 = torch.ops.aten.squeeze.dim(squeeze_dim_694, 0)
        squeeze_dim_694 = None
        unsqueeze_default_460 = torch.ops.aten.unsqueeze.default(primals_448, -1)
        unsqueeze_default_461 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_460, -1
        )
        unsqueeze_default_460 = None
        unsqueeze_default_462 = torch.ops.aten.unsqueeze.default(primals_449, -1)
        primals_449 = None
        unsqueeze_default_463 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_462, -1
        )
        unsqueeze_default_462 = None
        mul_tensor_811 = torch.ops.aten.mul.Tensor(
            mul_tensor_805, unsqueeze_default_461
        )
        mul_tensor_805 = unsqueeze_default_461 = None
        add_tensor_510 = torch.ops.aten.add.Tensor(
            mul_tensor_811, unsqueeze_default_463
        )
        mul_tensor_811 = unsqueeze_default_463 = None
        convolution_default_215 = torch.ops.aten.convolution.default(
            relu_default_111,
            primals_450,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_111 = None
        convolution_default_216 = torch.ops.aten.convolution.default(
            convolution_default_215,
            primals_451,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_232 = torch.ops.aten.var.correction(
            convolution_default_216, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_116 = torch.ops.aten.mean.dim(convolution_default_216, [0, 2, 3], True)
        add_tensor_511 = torch.ops.aten.add.Tensor(var_correction_232, 0.001)
        var_correction_232 = None
        sqrt_default_116 = torch.ops.aten.sqrt.default(add_tensor_511)
        add_tensor_511 = None
        reciprocal_default_116 = torch.ops.aten.reciprocal.default(sqrt_default_116)
        sqrt_default_116 = None
        sub_tensor_116 = torch.ops.aten.sub.Tensor(
            convolution_default_216, mean_dim_116
        )
        mul_tensor_812 = torch.ops.aten.mul.Tensor(
            sub_tensor_116, reciprocal_default_116
        )
        sub_tensor_116 = None
        squeeze_dim_696 = torch.ops.aten.squeeze.dim(mean_dim_116, 3)
        mean_dim_116 = None
        squeeze_dim_697 = torch.ops.aten.squeeze.dim(squeeze_dim_696, 2)
        squeeze_dim_696 = None
        squeeze_dim_698 = torch.ops.aten.squeeze.dim(squeeze_dim_697, 0)
        squeeze_dim_697 = None
        squeeze_dim_699 = torch.ops.aten.squeeze.dim(reciprocal_default_116, 3)
        reciprocal_default_116 = None
        squeeze_dim_700 = torch.ops.aten.squeeze.dim(squeeze_dim_699, 2)
        squeeze_dim_699 = None
        squeeze_dim_701 = torch.ops.aten.squeeze.dim(squeeze_dim_700, 0)
        squeeze_dim_700 = None
        unsqueeze_default_464 = torch.ops.aten.unsqueeze.default(primals_452, -1)
        unsqueeze_default_465 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_464, -1
        )
        unsqueeze_default_464 = None
        unsqueeze_default_466 = torch.ops.aten.unsqueeze.default(primals_453, -1)
        primals_453 = None
        unsqueeze_default_467 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_466, -1
        )
        unsqueeze_default_466 = None
        mul_tensor_818 = torch.ops.aten.mul.Tensor(
            mul_tensor_812, unsqueeze_default_465
        )
        mul_tensor_812 = unsqueeze_default_465 = None
        add_tensor_514 = torch.ops.aten.add.Tensor(
            mul_tensor_818, unsqueeze_default_467
        )
        mul_tensor_818 = unsqueeze_default_467 = None
        relu_default_116 = torch.ops.aten.relu.default(add_tensor_514)
        add_tensor_514 = None
        convolution_default_217 = torch.ops.aten.convolution.default(
            relu_default_116,
            primals_454,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_218 = torch.ops.aten.convolution.default(
            convolution_default_217,
            primals_455,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_234 = torch.ops.aten.var.correction(
            convolution_default_218, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_117 = torch.ops.aten.mean.dim(convolution_default_218, [0, 2, 3], True)
        add_tensor_515 = torch.ops.aten.add.Tensor(var_correction_234, 0.001)
        var_correction_234 = None
        sqrt_default_117 = torch.ops.aten.sqrt.default(add_tensor_515)
        add_tensor_515 = None
        reciprocal_default_117 = torch.ops.aten.reciprocal.default(sqrt_default_117)
        sqrt_default_117 = None
        sub_tensor_117 = torch.ops.aten.sub.Tensor(
            convolution_default_218, mean_dim_117
        )
        mul_tensor_819 = torch.ops.aten.mul.Tensor(
            sub_tensor_117, reciprocal_default_117
        )
        sub_tensor_117 = None
        squeeze_dim_702 = torch.ops.aten.squeeze.dim(mean_dim_117, 3)
        mean_dim_117 = None
        squeeze_dim_703 = torch.ops.aten.squeeze.dim(squeeze_dim_702, 2)
        squeeze_dim_702 = None
        squeeze_dim_704 = torch.ops.aten.squeeze.dim(squeeze_dim_703, 0)
        squeeze_dim_703 = None
        squeeze_dim_705 = torch.ops.aten.squeeze.dim(reciprocal_default_117, 3)
        reciprocal_default_117 = None
        squeeze_dim_706 = torch.ops.aten.squeeze.dim(squeeze_dim_705, 2)
        squeeze_dim_705 = None
        squeeze_dim_707 = torch.ops.aten.squeeze.dim(squeeze_dim_706, 0)
        squeeze_dim_706 = None
        unsqueeze_default_468 = torch.ops.aten.unsqueeze.default(primals_456, -1)
        unsqueeze_default_469 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_468, -1
        )
        unsqueeze_default_468 = None
        unsqueeze_default_470 = torch.ops.aten.unsqueeze.default(primals_457, -1)
        primals_457 = None
        unsqueeze_default_471 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_470, -1
        )
        unsqueeze_default_470 = None
        mul_tensor_825 = torch.ops.aten.mul.Tensor(
            mul_tensor_819, unsqueeze_default_469
        )
        mul_tensor_819 = unsqueeze_default_469 = None
        add_tensor_518 = torch.ops.aten.add.Tensor(
            mul_tensor_825, unsqueeze_default_471
        )
        mul_tensor_825 = unsqueeze_default_471 = None
        add_tensor_519 = torch.ops.aten.add.Tensor(add_tensor_510, add_tensor_518)
        add_tensor_510 = add_tensor_518 = None
        avg_pool2d_default_30 = torch.ops.aten.avg_pool2d.default(
            add_tensor_485, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_520 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_30, add_tensor_481
        )
        avg_pool2d_default_30 = None
        avg_pool2d_default_31 = torch.ops.aten.avg_pool2d.default(
            add_tensor_481, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_521 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_31, avg_pool2d_default_31
        )
        avg_pool2d_default_31 = None
        convolution_default_219 = torch.ops.aten.convolution.default(
            relu_default_109,
            primals_458,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_109 = None
        convolution_default_220 = torch.ops.aten.convolution.default(
            convolution_default_219,
            primals_459,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_236 = torch.ops.aten.var.correction(
            convolution_default_220, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_118 = torch.ops.aten.mean.dim(convolution_default_220, [0, 2, 3], True)
        add_tensor_522 = torch.ops.aten.add.Tensor(var_correction_236, 0.001)
        var_correction_236 = None
        sqrt_default_118 = torch.ops.aten.sqrt.default(add_tensor_522)
        add_tensor_522 = None
        reciprocal_default_118 = torch.ops.aten.reciprocal.default(sqrt_default_118)
        sqrt_default_118 = None
        sub_tensor_118 = torch.ops.aten.sub.Tensor(
            convolution_default_220, mean_dim_118
        )
        mul_tensor_826 = torch.ops.aten.mul.Tensor(
            sub_tensor_118, reciprocal_default_118
        )
        sub_tensor_118 = None
        squeeze_dim_708 = torch.ops.aten.squeeze.dim(mean_dim_118, 3)
        mean_dim_118 = None
        squeeze_dim_709 = torch.ops.aten.squeeze.dim(squeeze_dim_708, 2)
        squeeze_dim_708 = None
        squeeze_dim_710 = torch.ops.aten.squeeze.dim(squeeze_dim_709, 0)
        squeeze_dim_709 = None
        squeeze_dim_711 = torch.ops.aten.squeeze.dim(reciprocal_default_118, 3)
        reciprocal_default_118 = None
        squeeze_dim_712 = torch.ops.aten.squeeze.dim(squeeze_dim_711, 2)
        squeeze_dim_711 = None
        squeeze_dim_713 = torch.ops.aten.squeeze.dim(squeeze_dim_712, 0)
        squeeze_dim_712 = None
        unsqueeze_default_472 = torch.ops.aten.unsqueeze.default(primals_460, -1)
        unsqueeze_default_473 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_472, -1
        )
        unsqueeze_default_472 = None
        unsqueeze_default_474 = torch.ops.aten.unsqueeze.default(primals_461, -1)
        primals_461 = None
        unsqueeze_default_475 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_474, -1
        )
        unsqueeze_default_474 = None
        mul_tensor_832 = torch.ops.aten.mul.Tensor(
            mul_tensor_826, unsqueeze_default_473
        )
        mul_tensor_826 = unsqueeze_default_473 = None
        add_tensor_525 = torch.ops.aten.add.Tensor(
            mul_tensor_832, unsqueeze_default_475
        )
        mul_tensor_832 = unsqueeze_default_475 = None
        relu_default_118 = torch.ops.aten.relu.default(add_tensor_525)
        add_tensor_525 = None
        convolution_default_221 = torch.ops.aten.convolution.default(
            relu_default_118,
            primals_462,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_222 = torch.ops.aten.convolution.default(
            convolution_default_221,
            primals_463,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_238 = torch.ops.aten.var.correction(
            convolution_default_222, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_119 = torch.ops.aten.mean.dim(convolution_default_222, [0, 2, 3], True)
        add_tensor_526 = torch.ops.aten.add.Tensor(var_correction_238, 0.001)
        var_correction_238 = None
        sqrt_default_119 = torch.ops.aten.sqrt.default(add_tensor_526)
        add_tensor_526 = None
        reciprocal_default_119 = torch.ops.aten.reciprocal.default(sqrt_default_119)
        sqrt_default_119 = None
        sub_tensor_119 = torch.ops.aten.sub.Tensor(
            convolution_default_222, mean_dim_119
        )
        mul_tensor_833 = torch.ops.aten.mul.Tensor(
            sub_tensor_119, reciprocal_default_119
        )
        sub_tensor_119 = None
        squeeze_dim_714 = torch.ops.aten.squeeze.dim(mean_dim_119, 3)
        mean_dim_119 = None
        squeeze_dim_715 = torch.ops.aten.squeeze.dim(squeeze_dim_714, 2)
        squeeze_dim_714 = None
        squeeze_dim_716 = torch.ops.aten.squeeze.dim(squeeze_dim_715, 0)
        squeeze_dim_715 = None
        squeeze_dim_717 = torch.ops.aten.squeeze.dim(reciprocal_default_119, 3)
        reciprocal_default_119 = None
        squeeze_dim_718 = torch.ops.aten.squeeze.dim(squeeze_dim_717, 2)
        squeeze_dim_717 = None
        squeeze_dim_719 = torch.ops.aten.squeeze.dim(squeeze_dim_718, 0)
        squeeze_dim_718 = None
        unsqueeze_default_476 = torch.ops.aten.unsqueeze.default(primals_464, -1)
        unsqueeze_default_477 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_476, -1
        )
        unsqueeze_default_476 = None
        unsqueeze_default_478 = torch.ops.aten.unsqueeze.default(primals_465, -1)
        primals_465 = None
        unsqueeze_default_479 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_478, -1
        )
        unsqueeze_default_478 = None
        mul_tensor_839 = torch.ops.aten.mul.Tensor(
            mul_tensor_833, unsqueeze_default_477
        )
        mul_tensor_833 = unsqueeze_default_477 = None
        add_tensor_529 = torch.ops.aten.add.Tensor(
            mul_tensor_839, unsqueeze_default_479
        )
        mul_tensor_839 = unsqueeze_default_479 = None
        add_tensor_530 = torch.ops.aten.add.Tensor(add_tensor_529, add_tensor_485)
        add_tensor_529 = add_tensor_485 = None
        cat_default_12 = torch.ops.aten.cat.default(
            [
                add_tensor_481,
                add_tensor_502,
                add_tensor_519,
                add_tensor_520,
                add_tensor_521,
                add_tensor_530,
            ],
            1,
        )
        add_tensor_481 = (
            add_tensor_502
        ) = add_tensor_519 = add_tensor_520 = add_tensor_521 = add_tensor_530 = None
        convolution_default_223 = torch.ops.aten.convolution.default(
            relu_default_108,
            primals_466,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_240 = torch.ops.aten.var.correction(
            convolution_default_223, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_120 = torch.ops.aten.mean.dim(convolution_default_223, [0, 2, 3], True)
        add_tensor_531 = torch.ops.aten.add.Tensor(var_correction_240, 0.001)
        var_correction_240 = None
        sqrt_default_120 = torch.ops.aten.sqrt.default(add_tensor_531)
        add_tensor_531 = None
        reciprocal_default_120 = torch.ops.aten.reciprocal.default(sqrt_default_120)
        sqrt_default_120 = None
        sub_tensor_120 = torch.ops.aten.sub.Tensor(
            convolution_default_223, mean_dim_120
        )
        mul_tensor_840 = torch.ops.aten.mul.Tensor(
            sub_tensor_120, reciprocal_default_120
        )
        sub_tensor_120 = None
        unsqueeze_default_480 = torch.ops.aten.unsqueeze.default(primals_467, -1)
        unsqueeze_default_481 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_480, -1
        )
        unsqueeze_default_480 = None
        unsqueeze_default_482 = torch.ops.aten.unsqueeze.default(primals_468, -1)
        unsqueeze_default_483 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_482, -1
        )
        unsqueeze_default_482 = None
        mul_tensor_846 = torch.ops.aten.mul.Tensor(
            mul_tensor_840, unsqueeze_default_481
        )
        mul_tensor_840 = unsqueeze_default_481 = None
        add_tensor_534 = torch.ops.aten.add.Tensor(
            mul_tensor_846, unsqueeze_default_483
        )
        mul_tensor_846 = unsqueeze_default_483 = None
        relu_default_120 = torch.ops.aten.relu.default(cat_default_12)
        cat_default_12 = None
        convolution_default_224 = torch.ops.aten.convolution.default(
            relu_default_120,
            primals_469,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_242 = torch.ops.aten.var.correction(
            convolution_default_224, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_121 = torch.ops.aten.mean.dim(convolution_default_224, [0, 2, 3], True)
        add_tensor_535 = torch.ops.aten.add.Tensor(var_correction_242, 0.001)
        var_correction_242 = None
        sqrt_default_121 = torch.ops.aten.sqrt.default(add_tensor_535)
        add_tensor_535 = None
        reciprocal_default_121 = torch.ops.aten.reciprocal.default(sqrt_default_121)
        sqrt_default_121 = None
        sub_tensor_121 = torch.ops.aten.sub.Tensor(
            convolution_default_224, mean_dim_121
        )
        mul_tensor_847 = torch.ops.aten.mul.Tensor(
            sub_tensor_121, reciprocal_default_121
        )
        sub_tensor_121 = None
        unsqueeze_default_484 = torch.ops.aten.unsqueeze.default(primals_470, -1)
        unsqueeze_default_485 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_484, -1
        )
        unsqueeze_default_484 = None
        unsqueeze_default_486 = torch.ops.aten.unsqueeze.default(primals_471, -1)
        unsqueeze_default_487 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_486, -1
        )
        unsqueeze_default_486 = None
        mul_tensor_853 = torch.ops.aten.mul.Tensor(
            mul_tensor_847, unsqueeze_default_485
        )
        mul_tensor_847 = unsqueeze_default_485 = None
        add_tensor_538 = torch.ops.aten.add.Tensor(
            mul_tensor_853, unsqueeze_default_487
        )
        mul_tensor_853 = unsqueeze_default_487 = None
        relu_default_121 = torch.ops.aten.relu.default(add_tensor_538)
        convolution_default_225 = torch.ops.aten.convolution.default(
            relu_default_121,
            primals_472,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_226 = torch.ops.aten.convolution.default(
            convolution_default_225,
            primals_473,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_244 = torch.ops.aten.var.correction(
            convolution_default_226, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_122 = torch.ops.aten.mean.dim(convolution_default_226, [0, 2, 3], True)
        add_tensor_539 = torch.ops.aten.add.Tensor(var_correction_244, 0.001)
        var_correction_244 = None
        sqrt_default_122 = torch.ops.aten.sqrt.default(add_tensor_539)
        add_tensor_539 = None
        reciprocal_default_122 = torch.ops.aten.reciprocal.default(sqrt_default_122)
        sqrt_default_122 = None
        sub_tensor_122 = torch.ops.aten.sub.Tensor(
            convolution_default_226, mean_dim_122
        )
        mul_tensor_854 = torch.ops.aten.mul.Tensor(
            sub_tensor_122, reciprocal_default_122
        )
        sub_tensor_122 = None
        squeeze_dim_732 = torch.ops.aten.squeeze.dim(mean_dim_122, 3)
        mean_dim_122 = None
        squeeze_dim_733 = torch.ops.aten.squeeze.dim(squeeze_dim_732, 2)
        squeeze_dim_732 = None
        squeeze_dim_734 = torch.ops.aten.squeeze.dim(squeeze_dim_733, 0)
        squeeze_dim_733 = None
        squeeze_dim_735 = torch.ops.aten.squeeze.dim(reciprocal_default_122, 3)
        reciprocal_default_122 = None
        squeeze_dim_736 = torch.ops.aten.squeeze.dim(squeeze_dim_735, 2)
        squeeze_dim_735 = None
        squeeze_dim_737 = torch.ops.aten.squeeze.dim(squeeze_dim_736, 0)
        squeeze_dim_736 = None
        unsqueeze_default_488 = torch.ops.aten.unsqueeze.default(primals_474, -1)
        unsqueeze_default_489 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_488, -1
        )
        unsqueeze_default_488 = None
        unsqueeze_default_490 = torch.ops.aten.unsqueeze.default(primals_475, -1)
        primals_475 = None
        unsqueeze_default_491 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_490, -1
        )
        unsqueeze_default_490 = None
        mul_tensor_860 = torch.ops.aten.mul.Tensor(
            mul_tensor_854, unsqueeze_default_489
        )
        mul_tensor_854 = unsqueeze_default_489 = None
        add_tensor_542 = torch.ops.aten.add.Tensor(
            mul_tensor_860, unsqueeze_default_491
        )
        mul_tensor_860 = unsqueeze_default_491 = None
        relu_default_122 = torch.ops.aten.relu.default(add_tensor_542)
        add_tensor_542 = None
        convolution_default_227 = torch.ops.aten.convolution.default(
            relu_default_122,
            primals_476,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_228 = torch.ops.aten.convolution.default(
            convolution_default_227,
            primals_477,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_246 = torch.ops.aten.var.correction(
            convolution_default_228, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_123 = torch.ops.aten.mean.dim(convolution_default_228, [0, 2, 3], True)
        add_tensor_543 = torch.ops.aten.add.Tensor(var_correction_246, 0.001)
        var_correction_246 = None
        sqrt_default_123 = torch.ops.aten.sqrt.default(add_tensor_543)
        add_tensor_543 = None
        reciprocal_default_123 = torch.ops.aten.reciprocal.default(sqrt_default_123)
        sqrt_default_123 = None
        sub_tensor_123 = torch.ops.aten.sub.Tensor(
            convolution_default_228, mean_dim_123
        )
        mul_tensor_861 = torch.ops.aten.mul.Tensor(
            sub_tensor_123, reciprocal_default_123
        )
        sub_tensor_123 = None
        squeeze_dim_738 = torch.ops.aten.squeeze.dim(mean_dim_123, 3)
        mean_dim_123 = None
        squeeze_dim_739 = torch.ops.aten.squeeze.dim(squeeze_dim_738, 2)
        squeeze_dim_738 = None
        squeeze_dim_740 = torch.ops.aten.squeeze.dim(squeeze_dim_739, 0)
        squeeze_dim_739 = None
        squeeze_dim_741 = torch.ops.aten.squeeze.dim(reciprocal_default_123, 3)
        reciprocal_default_123 = None
        squeeze_dim_742 = torch.ops.aten.squeeze.dim(squeeze_dim_741, 2)
        squeeze_dim_741 = None
        squeeze_dim_743 = torch.ops.aten.squeeze.dim(squeeze_dim_742, 0)
        squeeze_dim_742 = None
        unsqueeze_default_492 = torch.ops.aten.unsqueeze.default(primals_478, -1)
        unsqueeze_default_493 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_492, -1
        )
        unsqueeze_default_492 = None
        unsqueeze_default_494 = torch.ops.aten.unsqueeze.default(primals_479, -1)
        primals_479 = None
        unsqueeze_default_495 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_494, -1
        )
        unsqueeze_default_494 = None
        mul_tensor_867 = torch.ops.aten.mul.Tensor(
            mul_tensor_861, unsqueeze_default_493
        )
        mul_tensor_861 = unsqueeze_default_493 = None
        add_tensor_546 = torch.ops.aten.add.Tensor(
            mul_tensor_867, unsqueeze_default_495
        )
        mul_tensor_867 = unsqueeze_default_495 = None
        relu_default_123 = torch.ops.aten.relu.default(add_tensor_534)
        convolution_default_229 = torch.ops.aten.convolution.default(
            relu_default_123,
            primals_480,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_230 = torch.ops.aten.convolution.default(
            convolution_default_229,
            primals_481,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_248 = torch.ops.aten.var.correction(
            convolution_default_230, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_124 = torch.ops.aten.mean.dim(convolution_default_230, [0, 2, 3], True)
        add_tensor_547 = torch.ops.aten.add.Tensor(var_correction_248, 0.001)
        var_correction_248 = None
        sqrt_default_124 = torch.ops.aten.sqrt.default(add_tensor_547)
        add_tensor_547 = None
        reciprocal_default_124 = torch.ops.aten.reciprocal.default(sqrt_default_124)
        sqrt_default_124 = None
        sub_tensor_124 = torch.ops.aten.sub.Tensor(
            convolution_default_230, mean_dim_124
        )
        mul_tensor_868 = torch.ops.aten.mul.Tensor(
            sub_tensor_124, reciprocal_default_124
        )
        sub_tensor_124 = None
        squeeze_dim_744 = torch.ops.aten.squeeze.dim(mean_dim_124, 3)
        mean_dim_124 = None
        squeeze_dim_745 = torch.ops.aten.squeeze.dim(squeeze_dim_744, 2)
        squeeze_dim_744 = None
        squeeze_dim_746 = torch.ops.aten.squeeze.dim(squeeze_dim_745, 0)
        squeeze_dim_745 = None
        squeeze_dim_747 = torch.ops.aten.squeeze.dim(reciprocal_default_124, 3)
        reciprocal_default_124 = None
        squeeze_dim_748 = torch.ops.aten.squeeze.dim(squeeze_dim_747, 2)
        squeeze_dim_747 = None
        squeeze_dim_749 = torch.ops.aten.squeeze.dim(squeeze_dim_748, 0)
        squeeze_dim_748 = None
        unsqueeze_default_496 = torch.ops.aten.unsqueeze.default(primals_482, -1)
        unsqueeze_default_497 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_496, -1
        )
        unsqueeze_default_496 = None
        unsqueeze_default_498 = torch.ops.aten.unsqueeze.default(primals_483, -1)
        primals_483 = None
        unsqueeze_default_499 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_498, -1
        )
        unsqueeze_default_498 = None
        mul_tensor_874 = torch.ops.aten.mul.Tensor(
            mul_tensor_868, unsqueeze_default_497
        )
        mul_tensor_868 = unsqueeze_default_497 = None
        add_tensor_550 = torch.ops.aten.add.Tensor(
            mul_tensor_874, unsqueeze_default_499
        )
        mul_tensor_874 = unsqueeze_default_499 = None
        relu_default_124 = torch.ops.aten.relu.default(add_tensor_550)
        add_tensor_550 = None
        convolution_default_231 = torch.ops.aten.convolution.default(
            relu_default_124,
            primals_484,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_232 = torch.ops.aten.convolution.default(
            convolution_default_231,
            primals_485,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_250 = torch.ops.aten.var.correction(
            convolution_default_232, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_125 = torch.ops.aten.mean.dim(convolution_default_232, [0, 2, 3], True)
        add_tensor_551 = torch.ops.aten.add.Tensor(var_correction_250, 0.001)
        var_correction_250 = None
        sqrt_default_125 = torch.ops.aten.sqrt.default(add_tensor_551)
        add_tensor_551 = None
        reciprocal_default_125 = torch.ops.aten.reciprocal.default(sqrt_default_125)
        sqrt_default_125 = None
        sub_tensor_125 = torch.ops.aten.sub.Tensor(
            convolution_default_232, mean_dim_125
        )
        mul_tensor_875 = torch.ops.aten.mul.Tensor(
            sub_tensor_125, reciprocal_default_125
        )
        sub_tensor_125 = None
        squeeze_dim_750 = torch.ops.aten.squeeze.dim(mean_dim_125, 3)
        mean_dim_125 = None
        squeeze_dim_751 = torch.ops.aten.squeeze.dim(squeeze_dim_750, 2)
        squeeze_dim_750 = None
        squeeze_dim_752 = torch.ops.aten.squeeze.dim(squeeze_dim_751, 0)
        squeeze_dim_751 = None
        squeeze_dim_753 = torch.ops.aten.squeeze.dim(reciprocal_default_125, 3)
        reciprocal_default_125 = None
        squeeze_dim_754 = torch.ops.aten.squeeze.dim(squeeze_dim_753, 2)
        squeeze_dim_753 = None
        squeeze_dim_755 = torch.ops.aten.squeeze.dim(squeeze_dim_754, 0)
        squeeze_dim_754 = None
        unsqueeze_default_500 = torch.ops.aten.unsqueeze.default(primals_486, -1)
        unsqueeze_default_501 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_500, -1
        )
        unsqueeze_default_500 = None
        unsqueeze_default_502 = torch.ops.aten.unsqueeze.default(primals_487, -1)
        primals_487 = None
        unsqueeze_default_503 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_502, -1
        )
        unsqueeze_default_502 = None
        mul_tensor_881 = torch.ops.aten.mul.Tensor(
            mul_tensor_875, unsqueeze_default_501
        )
        mul_tensor_875 = unsqueeze_default_501 = None
        add_tensor_554 = torch.ops.aten.add.Tensor(
            mul_tensor_881, unsqueeze_default_503
        )
        mul_tensor_881 = unsqueeze_default_503 = None
        add_tensor_555 = torch.ops.aten.add.Tensor(add_tensor_546, add_tensor_554)
        add_tensor_546 = add_tensor_554 = None
        convolution_default_233 = torch.ops.aten.convolution.default(
            relu_default_123,
            primals_488,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_234 = torch.ops.aten.convolution.default(
            convolution_default_233,
            primals_489,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_252 = torch.ops.aten.var.correction(
            convolution_default_234, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_126 = torch.ops.aten.mean.dim(convolution_default_234, [0, 2, 3], True)
        add_tensor_556 = torch.ops.aten.add.Tensor(var_correction_252, 0.001)
        var_correction_252 = None
        sqrt_default_126 = torch.ops.aten.sqrt.default(add_tensor_556)
        add_tensor_556 = None
        reciprocal_default_126 = torch.ops.aten.reciprocal.default(sqrt_default_126)
        sqrt_default_126 = None
        sub_tensor_126 = torch.ops.aten.sub.Tensor(
            convolution_default_234, mean_dim_126
        )
        mul_tensor_882 = torch.ops.aten.mul.Tensor(
            sub_tensor_126, reciprocal_default_126
        )
        sub_tensor_126 = None
        squeeze_dim_756 = torch.ops.aten.squeeze.dim(mean_dim_126, 3)
        mean_dim_126 = None
        squeeze_dim_757 = torch.ops.aten.squeeze.dim(squeeze_dim_756, 2)
        squeeze_dim_756 = None
        squeeze_dim_758 = torch.ops.aten.squeeze.dim(squeeze_dim_757, 0)
        squeeze_dim_757 = None
        squeeze_dim_759 = torch.ops.aten.squeeze.dim(reciprocal_default_126, 3)
        reciprocal_default_126 = None
        squeeze_dim_760 = torch.ops.aten.squeeze.dim(squeeze_dim_759, 2)
        squeeze_dim_759 = None
        squeeze_dim_761 = torch.ops.aten.squeeze.dim(squeeze_dim_760, 0)
        squeeze_dim_760 = None
        unsqueeze_default_504 = torch.ops.aten.unsqueeze.default(primals_490, -1)
        unsqueeze_default_505 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_504, -1
        )
        unsqueeze_default_504 = None
        unsqueeze_default_506 = torch.ops.aten.unsqueeze.default(primals_491, -1)
        primals_491 = None
        unsqueeze_default_507 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_506, -1
        )
        unsqueeze_default_506 = None
        mul_tensor_888 = torch.ops.aten.mul.Tensor(
            mul_tensor_882, unsqueeze_default_505
        )
        mul_tensor_882 = unsqueeze_default_505 = None
        add_tensor_559 = torch.ops.aten.add.Tensor(
            mul_tensor_888, unsqueeze_default_507
        )
        mul_tensor_888 = unsqueeze_default_507 = None
        relu_default_126 = torch.ops.aten.relu.default(add_tensor_559)
        add_tensor_559 = None
        convolution_default_235 = torch.ops.aten.convolution.default(
            relu_default_126,
            primals_492,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_236 = torch.ops.aten.convolution.default(
            convolution_default_235,
            primals_493,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_254 = torch.ops.aten.var.correction(
            convolution_default_236, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_127 = torch.ops.aten.mean.dim(convolution_default_236, [0, 2, 3], True)
        add_tensor_560 = torch.ops.aten.add.Tensor(var_correction_254, 0.001)
        var_correction_254 = None
        sqrt_default_127 = torch.ops.aten.sqrt.default(add_tensor_560)
        add_tensor_560 = None
        reciprocal_default_127 = torch.ops.aten.reciprocal.default(sqrt_default_127)
        sqrt_default_127 = None
        sub_tensor_127 = torch.ops.aten.sub.Tensor(
            convolution_default_236, mean_dim_127
        )
        mul_tensor_889 = torch.ops.aten.mul.Tensor(
            sub_tensor_127, reciprocal_default_127
        )
        sub_tensor_127 = None
        squeeze_dim_762 = torch.ops.aten.squeeze.dim(mean_dim_127, 3)
        mean_dim_127 = None
        squeeze_dim_763 = torch.ops.aten.squeeze.dim(squeeze_dim_762, 2)
        squeeze_dim_762 = None
        squeeze_dim_764 = torch.ops.aten.squeeze.dim(squeeze_dim_763, 0)
        squeeze_dim_763 = None
        squeeze_dim_765 = torch.ops.aten.squeeze.dim(reciprocal_default_127, 3)
        reciprocal_default_127 = None
        squeeze_dim_766 = torch.ops.aten.squeeze.dim(squeeze_dim_765, 2)
        squeeze_dim_765 = None
        squeeze_dim_767 = torch.ops.aten.squeeze.dim(squeeze_dim_766, 0)
        squeeze_dim_766 = None
        unsqueeze_default_508 = torch.ops.aten.unsqueeze.default(primals_494, -1)
        unsqueeze_default_509 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_508, -1
        )
        unsqueeze_default_508 = None
        unsqueeze_default_510 = torch.ops.aten.unsqueeze.default(primals_495, -1)
        primals_495 = None
        unsqueeze_default_511 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_510, -1
        )
        unsqueeze_default_510 = None
        mul_tensor_895 = torch.ops.aten.mul.Tensor(
            mul_tensor_889, unsqueeze_default_509
        )
        mul_tensor_889 = unsqueeze_default_509 = None
        add_tensor_563 = torch.ops.aten.add.Tensor(
            mul_tensor_895, unsqueeze_default_511
        )
        mul_tensor_895 = unsqueeze_default_511 = None
        convolution_default_237 = torch.ops.aten.convolution.default(
            relu_default_123,
            primals_496,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_123 = None
        convolution_default_238 = torch.ops.aten.convolution.default(
            convolution_default_237,
            primals_497,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_256 = torch.ops.aten.var.correction(
            convolution_default_238, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_128 = torch.ops.aten.mean.dim(convolution_default_238, [0, 2, 3], True)
        add_tensor_564 = torch.ops.aten.add.Tensor(var_correction_256, 0.001)
        var_correction_256 = None
        sqrt_default_128 = torch.ops.aten.sqrt.default(add_tensor_564)
        add_tensor_564 = None
        reciprocal_default_128 = torch.ops.aten.reciprocal.default(sqrt_default_128)
        sqrt_default_128 = None
        sub_tensor_128 = torch.ops.aten.sub.Tensor(
            convolution_default_238, mean_dim_128
        )
        mul_tensor_896 = torch.ops.aten.mul.Tensor(
            sub_tensor_128, reciprocal_default_128
        )
        sub_tensor_128 = None
        squeeze_dim_768 = torch.ops.aten.squeeze.dim(mean_dim_128, 3)
        mean_dim_128 = None
        squeeze_dim_769 = torch.ops.aten.squeeze.dim(squeeze_dim_768, 2)
        squeeze_dim_768 = None
        squeeze_dim_770 = torch.ops.aten.squeeze.dim(squeeze_dim_769, 0)
        squeeze_dim_769 = None
        squeeze_dim_771 = torch.ops.aten.squeeze.dim(reciprocal_default_128, 3)
        reciprocal_default_128 = None
        squeeze_dim_772 = torch.ops.aten.squeeze.dim(squeeze_dim_771, 2)
        squeeze_dim_771 = None
        squeeze_dim_773 = torch.ops.aten.squeeze.dim(squeeze_dim_772, 0)
        squeeze_dim_772 = None
        unsqueeze_default_512 = torch.ops.aten.unsqueeze.default(primals_498, -1)
        unsqueeze_default_513 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_512, -1
        )
        unsqueeze_default_512 = None
        unsqueeze_default_514 = torch.ops.aten.unsqueeze.default(primals_499, -1)
        primals_499 = None
        unsqueeze_default_515 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_514, -1
        )
        unsqueeze_default_514 = None
        mul_tensor_902 = torch.ops.aten.mul.Tensor(
            mul_tensor_896, unsqueeze_default_513
        )
        mul_tensor_896 = unsqueeze_default_513 = None
        add_tensor_567 = torch.ops.aten.add.Tensor(
            mul_tensor_902, unsqueeze_default_515
        )
        mul_tensor_902 = unsqueeze_default_515 = None
        relu_default_128 = torch.ops.aten.relu.default(add_tensor_567)
        add_tensor_567 = None
        convolution_default_239 = torch.ops.aten.convolution.default(
            relu_default_128,
            primals_500,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_240 = torch.ops.aten.convolution.default(
            convolution_default_239,
            primals_501,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_258 = torch.ops.aten.var.correction(
            convolution_default_240, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_129 = torch.ops.aten.mean.dim(convolution_default_240, [0, 2, 3], True)
        add_tensor_568 = torch.ops.aten.add.Tensor(var_correction_258, 0.001)
        var_correction_258 = None
        sqrt_default_129 = torch.ops.aten.sqrt.default(add_tensor_568)
        add_tensor_568 = None
        reciprocal_default_129 = torch.ops.aten.reciprocal.default(sqrt_default_129)
        sqrt_default_129 = None
        sub_tensor_129 = torch.ops.aten.sub.Tensor(
            convolution_default_240, mean_dim_129
        )
        mul_tensor_903 = torch.ops.aten.mul.Tensor(
            sub_tensor_129, reciprocal_default_129
        )
        sub_tensor_129 = None
        squeeze_dim_774 = torch.ops.aten.squeeze.dim(mean_dim_129, 3)
        mean_dim_129 = None
        squeeze_dim_775 = torch.ops.aten.squeeze.dim(squeeze_dim_774, 2)
        squeeze_dim_774 = None
        squeeze_dim_776 = torch.ops.aten.squeeze.dim(squeeze_dim_775, 0)
        squeeze_dim_775 = None
        squeeze_dim_777 = torch.ops.aten.squeeze.dim(reciprocal_default_129, 3)
        reciprocal_default_129 = None
        squeeze_dim_778 = torch.ops.aten.squeeze.dim(squeeze_dim_777, 2)
        squeeze_dim_777 = None
        squeeze_dim_779 = torch.ops.aten.squeeze.dim(squeeze_dim_778, 0)
        squeeze_dim_778 = None
        unsqueeze_default_516 = torch.ops.aten.unsqueeze.default(primals_502, -1)
        unsqueeze_default_517 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_516, -1
        )
        unsqueeze_default_516 = None
        unsqueeze_default_518 = torch.ops.aten.unsqueeze.default(primals_503, -1)
        primals_503 = None
        unsqueeze_default_519 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_518, -1
        )
        unsqueeze_default_518 = None
        mul_tensor_909 = torch.ops.aten.mul.Tensor(
            mul_tensor_903, unsqueeze_default_517
        )
        mul_tensor_903 = unsqueeze_default_517 = None
        add_tensor_571 = torch.ops.aten.add.Tensor(
            mul_tensor_909, unsqueeze_default_519
        )
        mul_tensor_909 = unsqueeze_default_519 = None
        add_tensor_572 = torch.ops.aten.add.Tensor(add_tensor_563, add_tensor_571)
        add_tensor_563 = add_tensor_571 = None
        avg_pool2d_default_33 = torch.ops.aten.avg_pool2d.default(
            add_tensor_538, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_573 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_33, add_tensor_534
        )
        avg_pool2d_default_33 = None
        avg_pool2d_default_34 = torch.ops.aten.avg_pool2d.default(
            add_tensor_534, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_574 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_34, avg_pool2d_default_34
        )
        avg_pool2d_default_34 = None
        convolution_default_241 = torch.ops.aten.convolution.default(
            relu_default_121,
            primals_504,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_121 = None
        convolution_default_242 = torch.ops.aten.convolution.default(
            convolution_default_241,
            primals_505,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_260 = torch.ops.aten.var.correction(
            convolution_default_242, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_130 = torch.ops.aten.mean.dim(convolution_default_242, [0, 2, 3], True)
        add_tensor_575 = torch.ops.aten.add.Tensor(var_correction_260, 0.001)
        var_correction_260 = None
        sqrt_default_130 = torch.ops.aten.sqrt.default(add_tensor_575)
        add_tensor_575 = None
        reciprocal_default_130 = torch.ops.aten.reciprocal.default(sqrt_default_130)
        sqrt_default_130 = None
        sub_tensor_130 = torch.ops.aten.sub.Tensor(
            convolution_default_242, mean_dim_130
        )
        mul_tensor_910 = torch.ops.aten.mul.Tensor(
            sub_tensor_130, reciprocal_default_130
        )
        sub_tensor_130 = None
        squeeze_dim_780 = torch.ops.aten.squeeze.dim(mean_dim_130, 3)
        mean_dim_130 = None
        squeeze_dim_781 = torch.ops.aten.squeeze.dim(squeeze_dim_780, 2)
        squeeze_dim_780 = None
        squeeze_dim_782 = torch.ops.aten.squeeze.dim(squeeze_dim_781, 0)
        squeeze_dim_781 = None
        squeeze_dim_783 = torch.ops.aten.squeeze.dim(reciprocal_default_130, 3)
        reciprocal_default_130 = None
        squeeze_dim_784 = torch.ops.aten.squeeze.dim(squeeze_dim_783, 2)
        squeeze_dim_783 = None
        squeeze_dim_785 = torch.ops.aten.squeeze.dim(squeeze_dim_784, 0)
        squeeze_dim_784 = None
        unsqueeze_default_520 = torch.ops.aten.unsqueeze.default(primals_506, -1)
        unsqueeze_default_521 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_520, -1
        )
        unsqueeze_default_520 = None
        unsqueeze_default_522 = torch.ops.aten.unsqueeze.default(primals_507, -1)
        primals_507 = None
        unsqueeze_default_523 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_522, -1
        )
        unsqueeze_default_522 = None
        mul_tensor_916 = torch.ops.aten.mul.Tensor(
            mul_tensor_910, unsqueeze_default_521
        )
        mul_tensor_910 = unsqueeze_default_521 = None
        add_tensor_578 = torch.ops.aten.add.Tensor(
            mul_tensor_916, unsqueeze_default_523
        )
        mul_tensor_916 = unsqueeze_default_523 = None
        relu_default_130 = torch.ops.aten.relu.default(add_tensor_578)
        add_tensor_578 = None
        convolution_default_243 = torch.ops.aten.convolution.default(
            relu_default_130,
            primals_508,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_244 = torch.ops.aten.convolution.default(
            convolution_default_243,
            primals_509,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_262 = torch.ops.aten.var.correction(
            convolution_default_244, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_131 = torch.ops.aten.mean.dim(convolution_default_244, [0, 2, 3], True)
        add_tensor_579 = torch.ops.aten.add.Tensor(var_correction_262, 0.001)
        var_correction_262 = None
        sqrt_default_131 = torch.ops.aten.sqrt.default(add_tensor_579)
        add_tensor_579 = None
        reciprocal_default_131 = torch.ops.aten.reciprocal.default(sqrt_default_131)
        sqrt_default_131 = None
        sub_tensor_131 = torch.ops.aten.sub.Tensor(
            convolution_default_244, mean_dim_131
        )
        mul_tensor_917 = torch.ops.aten.mul.Tensor(
            sub_tensor_131, reciprocal_default_131
        )
        sub_tensor_131 = None
        squeeze_dim_786 = torch.ops.aten.squeeze.dim(mean_dim_131, 3)
        mean_dim_131 = None
        squeeze_dim_787 = torch.ops.aten.squeeze.dim(squeeze_dim_786, 2)
        squeeze_dim_786 = None
        squeeze_dim_788 = torch.ops.aten.squeeze.dim(squeeze_dim_787, 0)
        squeeze_dim_787 = None
        squeeze_dim_789 = torch.ops.aten.squeeze.dim(reciprocal_default_131, 3)
        reciprocal_default_131 = None
        squeeze_dim_790 = torch.ops.aten.squeeze.dim(squeeze_dim_789, 2)
        squeeze_dim_789 = None
        squeeze_dim_791 = torch.ops.aten.squeeze.dim(squeeze_dim_790, 0)
        squeeze_dim_790 = None
        unsqueeze_default_524 = torch.ops.aten.unsqueeze.default(primals_510, -1)
        unsqueeze_default_525 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_524, -1
        )
        unsqueeze_default_524 = None
        unsqueeze_default_526 = torch.ops.aten.unsqueeze.default(primals_511, -1)
        primals_511 = None
        unsqueeze_default_527 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_526, -1
        )
        unsqueeze_default_526 = None
        mul_tensor_923 = torch.ops.aten.mul.Tensor(
            mul_tensor_917, unsqueeze_default_525
        )
        mul_tensor_917 = unsqueeze_default_525 = None
        add_tensor_582 = torch.ops.aten.add.Tensor(
            mul_tensor_923, unsqueeze_default_527
        )
        mul_tensor_923 = unsqueeze_default_527 = None
        add_tensor_583 = torch.ops.aten.add.Tensor(add_tensor_582, add_tensor_538)
        add_tensor_582 = add_tensor_538 = None
        cat_default_13 = torch.ops.aten.cat.default(
            [
                add_tensor_534,
                add_tensor_555,
                add_tensor_572,
                add_tensor_573,
                add_tensor_574,
                add_tensor_583,
            ],
            1,
        )
        add_tensor_534 = (
            add_tensor_555
        ) = add_tensor_572 = add_tensor_573 = add_tensor_574 = add_tensor_583 = None
        convolution_default_245 = torch.ops.aten.convolution.default(
            relu_default_120,
            primals_512,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_264 = torch.ops.aten.var.correction(
            convolution_default_245, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_132 = torch.ops.aten.mean.dim(convolution_default_245, [0, 2, 3], True)
        add_tensor_584 = torch.ops.aten.add.Tensor(var_correction_264, 0.001)
        var_correction_264 = None
        sqrt_default_132 = torch.ops.aten.sqrt.default(add_tensor_584)
        add_tensor_584 = None
        reciprocal_default_132 = torch.ops.aten.reciprocal.default(sqrt_default_132)
        sqrt_default_132 = None
        sub_tensor_132 = torch.ops.aten.sub.Tensor(
            convolution_default_245, mean_dim_132
        )
        mul_tensor_924 = torch.ops.aten.mul.Tensor(
            sub_tensor_132, reciprocal_default_132
        )
        sub_tensor_132 = None
        unsqueeze_default_528 = torch.ops.aten.unsqueeze.default(primals_513, -1)
        unsqueeze_default_529 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_528, -1
        )
        unsqueeze_default_528 = None
        unsqueeze_default_530 = torch.ops.aten.unsqueeze.default(primals_514, -1)
        unsqueeze_default_531 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_530, -1
        )
        unsqueeze_default_530 = None
        mul_tensor_930 = torch.ops.aten.mul.Tensor(
            mul_tensor_924, unsqueeze_default_529
        )
        mul_tensor_924 = unsqueeze_default_529 = None
        add_tensor_587 = torch.ops.aten.add.Tensor(
            mul_tensor_930, unsqueeze_default_531
        )
        mul_tensor_930 = unsqueeze_default_531 = None
        relu_default_132 = torch.ops.aten.relu.default(cat_default_13)
        cat_default_13 = None
        convolution_default_246 = torch.ops.aten.convolution.default(
            relu_default_132,
            primals_515,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_266 = torch.ops.aten.var.correction(
            convolution_default_246, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_133 = torch.ops.aten.mean.dim(convolution_default_246, [0, 2, 3], True)
        add_tensor_588 = torch.ops.aten.add.Tensor(var_correction_266, 0.001)
        var_correction_266 = None
        sqrt_default_133 = torch.ops.aten.sqrt.default(add_tensor_588)
        add_tensor_588 = None
        reciprocal_default_133 = torch.ops.aten.reciprocal.default(sqrt_default_133)
        sqrt_default_133 = None
        sub_tensor_133 = torch.ops.aten.sub.Tensor(
            convolution_default_246, mean_dim_133
        )
        mul_tensor_931 = torch.ops.aten.mul.Tensor(
            sub_tensor_133, reciprocal_default_133
        )
        sub_tensor_133 = None
        unsqueeze_default_532 = torch.ops.aten.unsqueeze.default(primals_516, -1)
        unsqueeze_default_533 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_532, -1
        )
        unsqueeze_default_532 = None
        unsqueeze_default_534 = torch.ops.aten.unsqueeze.default(primals_517, -1)
        unsqueeze_default_535 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_534, -1
        )
        unsqueeze_default_534 = None
        mul_tensor_937 = torch.ops.aten.mul.Tensor(
            mul_tensor_931, unsqueeze_default_533
        )
        mul_tensor_931 = unsqueeze_default_533 = None
        add_tensor_591 = torch.ops.aten.add.Tensor(
            mul_tensor_937, unsqueeze_default_535
        )
        mul_tensor_937 = unsqueeze_default_535 = None
        relu_default_133 = torch.ops.aten.relu.default(add_tensor_591)
        convolution_default_247 = torch.ops.aten.convolution.default(
            relu_default_133,
            primals_518,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_248 = torch.ops.aten.convolution.default(
            convolution_default_247,
            primals_519,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_268 = torch.ops.aten.var.correction(
            convolution_default_248, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_134 = torch.ops.aten.mean.dim(convolution_default_248, [0, 2, 3], True)
        add_tensor_592 = torch.ops.aten.add.Tensor(var_correction_268, 0.001)
        var_correction_268 = None
        sqrt_default_134 = torch.ops.aten.sqrt.default(add_tensor_592)
        add_tensor_592 = None
        reciprocal_default_134 = torch.ops.aten.reciprocal.default(sqrt_default_134)
        sqrt_default_134 = None
        sub_tensor_134 = torch.ops.aten.sub.Tensor(
            convolution_default_248, mean_dim_134
        )
        mul_tensor_938 = torch.ops.aten.mul.Tensor(
            sub_tensor_134, reciprocal_default_134
        )
        sub_tensor_134 = None
        squeeze_dim_804 = torch.ops.aten.squeeze.dim(mean_dim_134, 3)
        mean_dim_134 = None
        squeeze_dim_805 = torch.ops.aten.squeeze.dim(squeeze_dim_804, 2)
        squeeze_dim_804 = None
        squeeze_dim_806 = torch.ops.aten.squeeze.dim(squeeze_dim_805, 0)
        squeeze_dim_805 = None
        squeeze_dim_807 = torch.ops.aten.squeeze.dim(reciprocal_default_134, 3)
        reciprocal_default_134 = None
        squeeze_dim_808 = torch.ops.aten.squeeze.dim(squeeze_dim_807, 2)
        squeeze_dim_807 = None
        squeeze_dim_809 = torch.ops.aten.squeeze.dim(squeeze_dim_808, 0)
        squeeze_dim_808 = None
        unsqueeze_default_536 = torch.ops.aten.unsqueeze.default(primals_520, -1)
        unsqueeze_default_537 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_536, -1
        )
        unsqueeze_default_536 = None
        unsqueeze_default_538 = torch.ops.aten.unsqueeze.default(primals_521, -1)
        primals_521 = None
        unsqueeze_default_539 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_538, -1
        )
        unsqueeze_default_538 = None
        mul_tensor_944 = torch.ops.aten.mul.Tensor(
            mul_tensor_938, unsqueeze_default_537
        )
        mul_tensor_938 = unsqueeze_default_537 = None
        add_tensor_595 = torch.ops.aten.add.Tensor(
            mul_tensor_944, unsqueeze_default_539
        )
        mul_tensor_944 = unsqueeze_default_539 = None
        relu_default_134 = torch.ops.aten.relu.default(add_tensor_595)
        add_tensor_595 = None
        convolution_default_249 = torch.ops.aten.convolution.default(
            relu_default_134,
            primals_522,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_250 = torch.ops.aten.convolution.default(
            convolution_default_249,
            primals_523,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_270 = torch.ops.aten.var.correction(
            convolution_default_250, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_135 = torch.ops.aten.mean.dim(convolution_default_250, [0, 2, 3], True)
        add_tensor_596 = torch.ops.aten.add.Tensor(var_correction_270, 0.001)
        var_correction_270 = None
        sqrt_default_135 = torch.ops.aten.sqrt.default(add_tensor_596)
        add_tensor_596 = None
        reciprocal_default_135 = torch.ops.aten.reciprocal.default(sqrt_default_135)
        sqrt_default_135 = None
        sub_tensor_135 = torch.ops.aten.sub.Tensor(
            convolution_default_250, mean_dim_135
        )
        mul_tensor_945 = torch.ops.aten.mul.Tensor(
            sub_tensor_135, reciprocal_default_135
        )
        sub_tensor_135 = None
        squeeze_dim_810 = torch.ops.aten.squeeze.dim(mean_dim_135, 3)
        mean_dim_135 = None
        squeeze_dim_811 = torch.ops.aten.squeeze.dim(squeeze_dim_810, 2)
        squeeze_dim_810 = None
        squeeze_dim_812 = torch.ops.aten.squeeze.dim(squeeze_dim_811, 0)
        squeeze_dim_811 = None
        squeeze_dim_813 = torch.ops.aten.squeeze.dim(reciprocal_default_135, 3)
        reciprocal_default_135 = None
        squeeze_dim_814 = torch.ops.aten.squeeze.dim(squeeze_dim_813, 2)
        squeeze_dim_813 = None
        squeeze_dim_815 = torch.ops.aten.squeeze.dim(squeeze_dim_814, 0)
        squeeze_dim_814 = None
        unsqueeze_default_540 = torch.ops.aten.unsqueeze.default(primals_524, -1)
        unsqueeze_default_541 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_540, -1
        )
        unsqueeze_default_540 = None
        unsqueeze_default_542 = torch.ops.aten.unsqueeze.default(primals_525, -1)
        primals_525 = None
        unsqueeze_default_543 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_542, -1
        )
        unsqueeze_default_542 = None
        mul_tensor_951 = torch.ops.aten.mul.Tensor(
            mul_tensor_945, unsqueeze_default_541
        )
        mul_tensor_945 = unsqueeze_default_541 = None
        add_tensor_599 = torch.ops.aten.add.Tensor(
            mul_tensor_951, unsqueeze_default_543
        )
        mul_tensor_951 = unsqueeze_default_543 = None
        relu_default_135 = torch.ops.aten.relu.default(add_tensor_587)
        convolution_default_251 = torch.ops.aten.convolution.default(
            relu_default_135,
            primals_526,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_252 = torch.ops.aten.convolution.default(
            convolution_default_251,
            primals_527,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_272 = torch.ops.aten.var.correction(
            convolution_default_252, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_136 = torch.ops.aten.mean.dim(convolution_default_252, [0, 2, 3], True)
        add_tensor_600 = torch.ops.aten.add.Tensor(var_correction_272, 0.001)
        var_correction_272 = None
        sqrt_default_136 = torch.ops.aten.sqrt.default(add_tensor_600)
        add_tensor_600 = None
        reciprocal_default_136 = torch.ops.aten.reciprocal.default(sqrt_default_136)
        sqrt_default_136 = None
        sub_tensor_136 = torch.ops.aten.sub.Tensor(
            convolution_default_252, mean_dim_136
        )
        mul_tensor_952 = torch.ops.aten.mul.Tensor(
            sub_tensor_136, reciprocal_default_136
        )
        sub_tensor_136 = None
        squeeze_dim_816 = torch.ops.aten.squeeze.dim(mean_dim_136, 3)
        mean_dim_136 = None
        squeeze_dim_817 = torch.ops.aten.squeeze.dim(squeeze_dim_816, 2)
        squeeze_dim_816 = None
        squeeze_dim_818 = torch.ops.aten.squeeze.dim(squeeze_dim_817, 0)
        squeeze_dim_817 = None
        squeeze_dim_819 = torch.ops.aten.squeeze.dim(reciprocal_default_136, 3)
        reciprocal_default_136 = None
        squeeze_dim_820 = torch.ops.aten.squeeze.dim(squeeze_dim_819, 2)
        squeeze_dim_819 = None
        squeeze_dim_821 = torch.ops.aten.squeeze.dim(squeeze_dim_820, 0)
        squeeze_dim_820 = None
        unsqueeze_default_544 = torch.ops.aten.unsqueeze.default(primals_528, -1)
        unsqueeze_default_545 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_544, -1
        )
        unsqueeze_default_544 = None
        unsqueeze_default_546 = torch.ops.aten.unsqueeze.default(primals_529, -1)
        primals_529 = None
        unsqueeze_default_547 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_546, -1
        )
        unsqueeze_default_546 = None
        mul_tensor_958 = torch.ops.aten.mul.Tensor(
            mul_tensor_952, unsqueeze_default_545
        )
        mul_tensor_952 = unsqueeze_default_545 = None
        add_tensor_603 = torch.ops.aten.add.Tensor(
            mul_tensor_958, unsqueeze_default_547
        )
        mul_tensor_958 = unsqueeze_default_547 = None
        relu_default_136 = torch.ops.aten.relu.default(add_tensor_603)
        add_tensor_603 = None
        convolution_default_253 = torch.ops.aten.convolution.default(
            relu_default_136,
            primals_530,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_254 = torch.ops.aten.convolution.default(
            convolution_default_253,
            primals_531,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_274 = torch.ops.aten.var.correction(
            convolution_default_254, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_137 = torch.ops.aten.mean.dim(convolution_default_254, [0, 2, 3], True)
        add_tensor_604 = torch.ops.aten.add.Tensor(var_correction_274, 0.001)
        var_correction_274 = None
        sqrt_default_137 = torch.ops.aten.sqrt.default(add_tensor_604)
        add_tensor_604 = None
        reciprocal_default_137 = torch.ops.aten.reciprocal.default(sqrt_default_137)
        sqrt_default_137 = None
        sub_tensor_137 = torch.ops.aten.sub.Tensor(
            convolution_default_254, mean_dim_137
        )
        mul_tensor_959 = torch.ops.aten.mul.Tensor(
            sub_tensor_137, reciprocal_default_137
        )
        sub_tensor_137 = None
        squeeze_dim_822 = torch.ops.aten.squeeze.dim(mean_dim_137, 3)
        mean_dim_137 = None
        squeeze_dim_823 = torch.ops.aten.squeeze.dim(squeeze_dim_822, 2)
        squeeze_dim_822 = None
        squeeze_dim_824 = torch.ops.aten.squeeze.dim(squeeze_dim_823, 0)
        squeeze_dim_823 = None
        squeeze_dim_825 = torch.ops.aten.squeeze.dim(reciprocal_default_137, 3)
        reciprocal_default_137 = None
        squeeze_dim_826 = torch.ops.aten.squeeze.dim(squeeze_dim_825, 2)
        squeeze_dim_825 = None
        squeeze_dim_827 = torch.ops.aten.squeeze.dim(squeeze_dim_826, 0)
        squeeze_dim_826 = None
        unsqueeze_default_548 = torch.ops.aten.unsqueeze.default(primals_532, -1)
        unsqueeze_default_549 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_548, -1
        )
        unsqueeze_default_548 = None
        unsqueeze_default_550 = torch.ops.aten.unsqueeze.default(primals_533, -1)
        primals_533 = None
        unsqueeze_default_551 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_550, -1
        )
        unsqueeze_default_550 = None
        mul_tensor_965 = torch.ops.aten.mul.Tensor(
            mul_tensor_959, unsqueeze_default_549
        )
        mul_tensor_959 = unsqueeze_default_549 = None
        add_tensor_607 = torch.ops.aten.add.Tensor(
            mul_tensor_965, unsqueeze_default_551
        )
        mul_tensor_965 = unsqueeze_default_551 = None
        add_tensor_608 = torch.ops.aten.add.Tensor(add_tensor_599, add_tensor_607)
        add_tensor_599 = add_tensor_607 = None
        convolution_default_255 = torch.ops.aten.convolution.default(
            relu_default_135,
            primals_534,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_256 = torch.ops.aten.convolution.default(
            convolution_default_255,
            primals_535,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_276 = torch.ops.aten.var.correction(
            convolution_default_256, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_138 = torch.ops.aten.mean.dim(convolution_default_256, [0, 2, 3], True)
        add_tensor_609 = torch.ops.aten.add.Tensor(var_correction_276, 0.001)
        var_correction_276 = None
        sqrt_default_138 = torch.ops.aten.sqrt.default(add_tensor_609)
        add_tensor_609 = None
        reciprocal_default_138 = torch.ops.aten.reciprocal.default(sqrt_default_138)
        sqrt_default_138 = None
        sub_tensor_138 = torch.ops.aten.sub.Tensor(
            convolution_default_256, mean_dim_138
        )
        mul_tensor_966 = torch.ops.aten.mul.Tensor(
            sub_tensor_138, reciprocal_default_138
        )
        sub_tensor_138 = None
        squeeze_dim_828 = torch.ops.aten.squeeze.dim(mean_dim_138, 3)
        mean_dim_138 = None
        squeeze_dim_829 = torch.ops.aten.squeeze.dim(squeeze_dim_828, 2)
        squeeze_dim_828 = None
        squeeze_dim_830 = torch.ops.aten.squeeze.dim(squeeze_dim_829, 0)
        squeeze_dim_829 = None
        squeeze_dim_831 = torch.ops.aten.squeeze.dim(reciprocal_default_138, 3)
        reciprocal_default_138 = None
        squeeze_dim_832 = torch.ops.aten.squeeze.dim(squeeze_dim_831, 2)
        squeeze_dim_831 = None
        squeeze_dim_833 = torch.ops.aten.squeeze.dim(squeeze_dim_832, 0)
        squeeze_dim_832 = None
        unsqueeze_default_552 = torch.ops.aten.unsqueeze.default(primals_536, -1)
        unsqueeze_default_553 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_552, -1
        )
        unsqueeze_default_552 = None
        unsqueeze_default_554 = torch.ops.aten.unsqueeze.default(primals_537, -1)
        primals_537 = None
        unsqueeze_default_555 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_554, -1
        )
        unsqueeze_default_554 = None
        mul_tensor_972 = torch.ops.aten.mul.Tensor(
            mul_tensor_966, unsqueeze_default_553
        )
        mul_tensor_966 = unsqueeze_default_553 = None
        add_tensor_612 = torch.ops.aten.add.Tensor(
            mul_tensor_972, unsqueeze_default_555
        )
        mul_tensor_972 = unsqueeze_default_555 = None
        relu_default_138 = torch.ops.aten.relu.default(add_tensor_612)
        add_tensor_612 = None
        convolution_default_257 = torch.ops.aten.convolution.default(
            relu_default_138,
            primals_538,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_258 = torch.ops.aten.convolution.default(
            convolution_default_257,
            primals_539,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_278 = torch.ops.aten.var.correction(
            convolution_default_258, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_139 = torch.ops.aten.mean.dim(convolution_default_258, [0, 2, 3], True)
        add_tensor_613 = torch.ops.aten.add.Tensor(var_correction_278, 0.001)
        var_correction_278 = None
        sqrt_default_139 = torch.ops.aten.sqrt.default(add_tensor_613)
        add_tensor_613 = None
        reciprocal_default_139 = torch.ops.aten.reciprocal.default(sqrt_default_139)
        sqrt_default_139 = None
        sub_tensor_139 = torch.ops.aten.sub.Tensor(
            convolution_default_258, mean_dim_139
        )
        mul_tensor_973 = torch.ops.aten.mul.Tensor(
            sub_tensor_139, reciprocal_default_139
        )
        sub_tensor_139 = None
        squeeze_dim_834 = torch.ops.aten.squeeze.dim(mean_dim_139, 3)
        mean_dim_139 = None
        squeeze_dim_835 = torch.ops.aten.squeeze.dim(squeeze_dim_834, 2)
        squeeze_dim_834 = None
        squeeze_dim_836 = torch.ops.aten.squeeze.dim(squeeze_dim_835, 0)
        squeeze_dim_835 = None
        squeeze_dim_837 = torch.ops.aten.squeeze.dim(reciprocal_default_139, 3)
        reciprocal_default_139 = None
        squeeze_dim_838 = torch.ops.aten.squeeze.dim(squeeze_dim_837, 2)
        squeeze_dim_837 = None
        squeeze_dim_839 = torch.ops.aten.squeeze.dim(squeeze_dim_838, 0)
        squeeze_dim_838 = None
        unsqueeze_default_556 = torch.ops.aten.unsqueeze.default(primals_540, -1)
        unsqueeze_default_557 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_556, -1
        )
        unsqueeze_default_556 = None
        unsqueeze_default_558 = torch.ops.aten.unsqueeze.default(primals_541, -1)
        primals_541 = None
        unsqueeze_default_559 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_558, -1
        )
        unsqueeze_default_558 = None
        mul_tensor_979 = torch.ops.aten.mul.Tensor(
            mul_tensor_973, unsqueeze_default_557
        )
        mul_tensor_973 = unsqueeze_default_557 = None
        add_tensor_616 = torch.ops.aten.add.Tensor(
            mul_tensor_979, unsqueeze_default_559
        )
        mul_tensor_979 = unsqueeze_default_559 = None
        convolution_default_259 = torch.ops.aten.convolution.default(
            relu_default_135,
            primals_542,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_135 = None
        convolution_default_260 = torch.ops.aten.convolution.default(
            convolution_default_259,
            primals_543,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_280 = torch.ops.aten.var.correction(
            convolution_default_260, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_140 = torch.ops.aten.mean.dim(convolution_default_260, [0, 2, 3], True)
        add_tensor_617 = torch.ops.aten.add.Tensor(var_correction_280, 0.001)
        var_correction_280 = None
        sqrt_default_140 = torch.ops.aten.sqrt.default(add_tensor_617)
        add_tensor_617 = None
        reciprocal_default_140 = torch.ops.aten.reciprocal.default(sqrt_default_140)
        sqrt_default_140 = None
        sub_tensor_140 = torch.ops.aten.sub.Tensor(
            convolution_default_260, mean_dim_140
        )
        mul_tensor_980 = torch.ops.aten.mul.Tensor(
            sub_tensor_140, reciprocal_default_140
        )
        sub_tensor_140 = None
        squeeze_dim_840 = torch.ops.aten.squeeze.dim(mean_dim_140, 3)
        mean_dim_140 = None
        squeeze_dim_841 = torch.ops.aten.squeeze.dim(squeeze_dim_840, 2)
        squeeze_dim_840 = None
        squeeze_dim_842 = torch.ops.aten.squeeze.dim(squeeze_dim_841, 0)
        squeeze_dim_841 = None
        squeeze_dim_843 = torch.ops.aten.squeeze.dim(reciprocal_default_140, 3)
        reciprocal_default_140 = None
        squeeze_dim_844 = torch.ops.aten.squeeze.dim(squeeze_dim_843, 2)
        squeeze_dim_843 = None
        squeeze_dim_845 = torch.ops.aten.squeeze.dim(squeeze_dim_844, 0)
        squeeze_dim_844 = None
        unsqueeze_default_560 = torch.ops.aten.unsqueeze.default(primals_544, -1)
        unsqueeze_default_561 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_560, -1
        )
        unsqueeze_default_560 = None
        unsqueeze_default_562 = torch.ops.aten.unsqueeze.default(primals_545, -1)
        primals_545 = None
        unsqueeze_default_563 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_562, -1
        )
        unsqueeze_default_562 = None
        mul_tensor_986 = torch.ops.aten.mul.Tensor(
            mul_tensor_980, unsqueeze_default_561
        )
        mul_tensor_980 = unsqueeze_default_561 = None
        add_tensor_620 = torch.ops.aten.add.Tensor(
            mul_tensor_986, unsqueeze_default_563
        )
        mul_tensor_986 = unsqueeze_default_563 = None
        relu_default_140 = torch.ops.aten.relu.default(add_tensor_620)
        add_tensor_620 = None
        convolution_default_261 = torch.ops.aten.convolution.default(
            relu_default_140,
            primals_546,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_262 = torch.ops.aten.convolution.default(
            convolution_default_261,
            primals_547,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_282 = torch.ops.aten.var.correction(
            convolution_default_262, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_141 = torch.ops.aten.mean.dim(convolution_default_262, [0, 2, 3], True)
        add_tensor_621 = torch.ops.aten.add.Tensor(var_correction_282, 0.001)
        var_correction_282 = None
        sqrt_default_141 = torch.ops.aten.sqrt.default(add_tensor_621)
        add_tensor_621 = None
        reciprocal_default_141 = torch.ops.aten.reciprocal.default(sqrt_default_141)
        sqrt_default_141 = None
        sub_tensor_141 = torch.ops.aten.sub.Tensor(
            convolution_default_262, mean_dim_141
        )
        mul_tensor_987 = torch.ops.aten.mul.Tensor(
            sub_tensor_141, reciprocal_default_141
        )
        sub_tensor_141 = None
        squeeze_dim_846 = torch.ops.aten.squeeze.dim(mean_dim_141, 3)
        mean_dim_141 = None
        squeeze_dim_847 = torch.ops.aten.squeeze.dim(squeeze_dim_846, 2)
        squeeze_dim_846 = None
        squeeze_dim_848 = torch.ops.aten.squeeze.dim(squeeze_dim_847, 0)
        squeeze_dim_847 = None
        squeeze_dim_849 = torch.ops.aten.squeeze.dim(reciprocal_default_141, 3)
        reciprocal_default_141 = None
        squeeze_dim_850 = torch.ops.aten.squeeze.dim(squeeze_dim_849, 2)
        squeeze_dim_849 = None
        squeeze_dim_851 = torch.ops.aten.squeeze.dim(squeeze_dim_850, 0)
        squeeze_dim_850 = None
        unsqueeze_default_564 = torch.ops.aten.unsqueeze.default(primals_548, -1)
        unsqueeze_default_565 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_564, -1
        )
        unsqueeze_default_564 = None
        unsqueeze_default_566 = torch.ops.aten.unsqueeze.default(primals_549, -1)
        primals_549 = None
        unsqueeze_default_567 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_566, -1
        )
        unsqueeze_default_566 = None
        mul_tensor_993 = torch.ops.aten.mul.Tensor(
            mul_tensor_987, unsqueeze_default_565
        )
        mul_tensor_987 = unsqueeze_default_565 = None
        add_tensor_624 = torch.ops.aten.add.Tensor(
            mul_tensor_993, unsqueeze_default_567
        )
        mul_tensor_993 = unsqueeze_default_567 = None
        add_tensor_625 = torch.ops.aten.add.Tensor(add_tensor_616, add_tensor_624)
        add_tensor_616 = add_tensor_624 = None
        avg_pool2d_default_36 = torch.ops.aten.avg_pool2d.default(
            add_tensor_591, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_626 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_36, add_tensor_587
        )
        avg_pool2d_default_36 = None
        avg_pool2d_default_37 = torch.ops.aten.avg_pool2d.default(
            add_tensor_587, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_627 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_37, avg_pool2d_default_37
        )
        avg_pool2d_default_37 = None
        convolution_default_263 = torch.ops.aten.convolution.default(
            relu_default_133,
            primals_550,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_133 = None
        convolution_default_264 = torch.ops.aten.convolution.default(
            convolution_default_263,
            primals_551,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_284 = torch.ops.aten.var.correction(
            convolution_default_264, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_142 = torch.ops.aten.mean.dim(convolution_default_264, [0, 2, 3], True)
        add_tensor_628 = torch.ops.aten.add.Tensor(var_correction_284, 0.001)
        var_correction_284 = None
        sqrt_default_142 = torch.ops.aten.sqrt.default(add_tensor_628)
        add_tensor_628 = None
        reciprocal_default_142 = torch.ops.aten.reciprocal.default(sqrt_default_142)
        sqrt_default_142 = None
        sub_tensor_142 = torch.ops.aten.sub.Tensor(
            convolution_default_264, mean_dim_142
        )
        mul_tensor_994 = torch.ops.aten.mul.Tensor(
            sub_tensor_142, reciprocal_default_142
        )
        sub_tensor_142 = None
        squeeze_dim_852 = torch.ops.aten.squeeze.dim(mean_dim_142, 3)
        mean_dim_142 = None
        squeeze_dim_853 = torch.ops.aten.squeeze.dim(squeeze_dim_852, 2)
        squeeze_dim_852 = None
        squeeze_dim_854 = torch.ops.aten.squeeze.dim(squeeze_dim_853, 0)
        squeeze_dim_853 = None
        squeeze_dim_855 = torch.ops.aten.squeeze.dim(reciprocal_default_142, 3)
        reciprocal_default_142 = None
        squeeze_dim_856 = torch.ops.aten.squeeze.dim(squeeze_dim_855, 2)
        squeeze_dim_855 = None
        squeeze_dim_857 = torch.ops.aten.squeeze.dim(squeeze_dim_856, 0)
        squeeze_dim_856 = None
        unsqueeze_default_568 = torch.ops.aten.unsqueeze.default(primals_552, -1)
        unsqueeze_default_569 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_568, -1
        )
        unsqueeze_default_568 = None
        unsqueeze_default_570 = torch.ops.aten.unsqueeze.default(primals_553, -1)
        primals_553 = None
        unsqueeze_default_571 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_570, -1
        )
        unsqueeze_default_570 = None
        mul_tensor_1000 = torch.ops.aten.mul.Tensor(
            mul_tensor_994, unsqueeze_default_569
        )
        mul_tensor_994 = unsqueeze_default_569 = None
        add_tensor_631 = torch.ops.aten.add.Tensor(
            mul_tensor_1000, unsqueeze_default_571
        )
        mul_tensor_1000 = unsqueeze_default_571 = None
        relu_default_142 = torch.ops.aten.relu.default(add_tensor_631)
        add_tensor_631 = None
        convolution_default_265 = torch.ops.aten.convolution.default(
            relu_default_142,
            primals_554,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_266 = torch.ops.aten.convolution.default(
            convolution_default_265,
            primals_555,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_286 = torch.ops.aten.var.correction(
            convolution_default_266, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_143 = torch.ops.aten.mean.dim(convolution_default_266, [0, 2, 3], True)
        add_tensor_632 = torch.ops.aten.add.Tensor(var_correction_286, 0.001)
        var_correction_286 = None
        sqrt_default_143 = torch.ops.aten.sqrt.default(add_tensor_632)
        add_tensor_632 = None
        reciprocal_default_143 = torch.ops.aten.reciprocal.default(sqrt_default_143)
        sqrt_default_143 = None
        sub_tensor_143 = torch.ops.aten.sub.Tensor(
            convolution_default_266, mean_dim_143
        )
        mul_tensor_1001 = torch.ops.aten.mul.Tensor(
            sub_tensor_143, reciprocal_default_143
        )
        sub_tensor_143 = None
        squeeze_dim_858 = torch.ops.aten.squeeze.dim(mean_dim_143, 3)
        mean_dim_143 = None
        squeeze_dim_859 = torch.ops.aten.squeeze.dim(squeeze_dim_858, 2)
        squeeze_dim_858 = None
        squeeze_dim_860 = torch.ops.aten.squeeze.dim(squeeze_dim_859, 0)
        squeeze_dim_859 = None
        squeeze_dim_861 = torch.ops.aten.squeeze.dim(reciprocal_default_143, 3)
        reciprocal_default_143 = None
        squeeze_dim_862 = torch.ops.aten.squeeze.dim(squeeze_dim_861, 2)
        squeeze_dim_861 = None
        squeeze_dim_863 = torch.ops.aten.squeeze.dim(squeeze_dim_862, 0)
        squeeze_dim_862 = None
        unsqueeze_default_572 = torch.ops.aten.unsqueeze.default(primals_556, -1)
        unsqueeze_default_573 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_572, -1
        )
        unsqueeze_default_572 = None
        unsqueeze_default_574 = torch.ops.aten.unsqueeze.default(primals_557, -1)
        primals_557 = None
        unsqueeze_default_575 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_574, -1
        )
        unsqueeze_default_574 = None
        mul_tensor_1007 = torch.ops.aten.mul.Tensor(
            mul_tensor_1001, unsqueeze_default_573
        )
        mul_tensor_1001 = unsqueeze_default_573 = None
        add_tensor_635 = torch.ops.aten.add.Tensor(
            mul_tensor_1007, unsqueeze_default_575
        )
        mul_tensor_1007 = unsqueeze_default_575 = None
        add_tensor_636 = torch.ops.aten.add.Tensor(add_tensor_635, add_tensor_591)
        add_tensor_635 = add_tensor_591 = None
        cat_default_14 = torch.ops.aten.cat.default(
            [
                add_tensor_587,
                add_tensor_608,
                add_tensor_625,
                add_tensor_626,
                add_tensor_627,
                add_tensor_636,
            ],
            1,
        )
        add_tensor_587 = (
            add_tensor_608
        ) = add_tensor_625 = add_tensor_626 = add_tensor_627 = add_tensor_636 = None
        convolution_default_267 = torch.ops.aten.convolution.default(
            relu_default_132,
            primals_558,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_288 = torch.ops.aten.var.correction(
            convolution_default_267, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_144 = torch.ops.aten.mean.dim(convolution_default_267, [0, 2, 3], True)
        add_tensor_637 = torch.ops.aten.add.Tensor(var_correction_288, 0.001)
        var_correction_288 = None
        sqrt_default_144 = torch.ops.aten.sqrt.default(add_tensor_637)
        add_tensor_637 = None
        reciprocal_default_144 = torch.ops.aten.reciprocal.default(sqrt_default_144)
        sqrt_default_144 = None
        sub_tensor_144 = torch.ops.aten.sub.Tensor(
            convolution_default_267, mean_dim_144
        )
        mul_tensor_1008 = torch.ops.aten.mul.Tensor(
            sub_tensor_144, reciprocal_default_144
        )
        sub_tensor_144 = None
        unsqueeze_default_576 = torch.ops.aten.unsqueeze.default(primals_559, -1)
        unsqueeze_default_577 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_576, -1
        )
        unsqueeze_default_576 = None
        unsqueeze_default_578 = torch.ops.aten.unsqueeze.default(primals_560, -1)
        unsqueeze_default_579 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_578, -1
        )
        unsqueeze_default_578 = None
        mul_tensor_1014 = torch.ops.aten.mul.Tensor(
            mul_tensor_1008, unsqueeze_default_577
        )
        mul_tensor_1008 = unsqueeze_default_577 = None
        add_tensor_640 = torch.ops.aten.add.Tensor(
            mul_tensor_1014, unsqueeze_default_579
        )
        mul_tensor_1014 = unsqueeze_default_579 = None
        relu_default_144 = torch.ops.aten.relu.default(cat_default_14)
        cat_default_14 = None
        convolution_default_268 = torch.ops.aten.convolution.default(
            relu_default_144,
            primals_561,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_290 = torch.ops.aten.var.correction(
            convolution_default_268, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_145 = torch.ops.aten.mean.dim(convolution_default_268, [0, 2, 3], True)
        add_tensor_641 = torch.ops.aten.add.Tensor(var_correction_290, 0.001)
        var_correction_290 = None
        sqrt_default_145 = torch.ops.aten.sqrt.default(add_tensor_641)
        add_tensor_641 = None
        reciprocal_default_145 = torch.ops.aten.reciprocal.default(sqrt_default_145)
        sqrt_default_145 = None
        sub_tensor_145 = torch.ops.aten.sub.Tensor(
            convolution_default_268, mean_dim_145
        )
        mul_tensor_1015 = torch.ops.aten.mul.Tensor(
            sub_tensor_145, reciprocal_default_145
        )
        sub_tensor_145 = None
        unsqueeze_default_580 = torch.ops.aten.unsqueeze.default(primals_562, -1)
        unsqueeze_default_581 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_580, -1
        )
        unsqueeze_default_580 = None
        unsqueeze_default_582 = torch.ops.aten.unsqueeze.default(primals_563, -1)
        unsqueeze_default_583 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_582, -1
        )
        unsqueeze_default_582 = None
        mul_tensor_1021 = torch.ops.aten.mul.Tensor(
            mul_tensor_1015, unsqueeze_default_581
        )
        mul_tensor_1015 = unsqueeze_default_581 = None
        add_tensor_644 = torch.ops.aten.add.Tensor(
            mul_tensor_1021, unsqueeze_default_583
        )
        mul_tensor_1021 = unsqueeze_default_583 = None
        relu_default_145 = torch.ops.aten.relu.default(add_tensor_644)
        convolution_default_269 = torch.ops.aten.convolution.default(
            relu_default_145,
            primals_564,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_270 = torch.ops.aten.convolution.default(
            convolution_default_269,
            primals_565,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_292 = torch.ops.aten.var.correction(
            convolution_default_270, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_146 = torch.ops.aten.mean.dim(convolution_default_270, [0, 2, 3], True)
        add_tensor_645 = torch.ops.aten.add.Tensor(var_correction_292, 0.001)
        var_correction_292 = None
        sqrt_default_146 = torch.ops.aten.sqrt.default(add_tensor_645)
        add_tensor_645 = None
        reciprocal_default_146 = torch.ops.aten.reciprocal.default(sqrt_default_146)
        sqrt_default_146 = None
        sub_tensor_146 = torch.ops.aten.sub.Tensor(
            convolution_default_270, mean_dim_146
        )
        mul_tensor_1022 = torch.ops.aten.mul.Tensor(
            sub_tensor_146, reciprocal_default_146
        )
        sub_tensor_146 = None
        squeeze_dim_876 = torch.ops.aten.squeeze.dim(mean_dim_146, 3)
        mean_dim_146 = None
        squeeze_dim_877 = torch.ops.aten.squeeze.dim(squeeze_dim_876, 2)
        squeeze_dim_876 = None
        squeeze_dim_878 = torch.ops.aten.squeeze.dim(squeeze_dim_877, 0)
        squeeze_dim_877 = None
        squeeze_dim_879 = torch.ops.aten.squeeze.dim(reciprocal_default_146, 3)
        reciprocal_default_146 = None
        squeeze_dim_880 = torch.ops.aten.squeeze.dim(squeeze_dim_879, 2)
        squeeze_dim_879 = None
        squeeze_dim_881 = torch.ops.aten.squeeze.dim(squeeze_dim_880, 0)
        squeeze_dim_880 = None
        unsqueeze_default_584 = torch.ops.aten.unsqueeze.default(primals_566, -1)
        unsqueeze_default_585 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_584, -1
        )
        unsqueeze_default_584 = None
        unsqueeze_default_586 = torch.ops.aten.unsqueeze.default(primals_567, -1)
        primals_567 = None
        unsqueeze_default_587 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_586, -1
        )
        unsqueeze_default_586 = None
        mul_tensor_1028 = torch.ops.aten.mul.Tensor(
            mul_tensor_1022, unsqueeze_default_585
        )
        mul_tensor_1022 = unsqueeze_default_585 = None
        add_tensor_648 = torch.ops.aten.add.Tensor(
            mul_tensor_1028, unsqueeze_default_587
        )
        mul_tensor_1028 = unsqueeze_default_587 = None
        relu_default_146 = torch.ops.aten.relu.default(add_tensor_648)
        add_tensor_648 = None
        convolution_default_271 = torch.ops.aten.convolution.default(
            relu_default_146,
            primals_568,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_272 = torch.ops.aten.convolution.default(
            convolution_default_271,
            primals_569,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_294 = torch.ops.aten.var.correction(
            convolution_default_272, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_147 = torch.ops.aten.mean.dim(convolution_default_272, [0, 2, 3], True)
        add_tensor_649 = torch.ops.aten.add.Tensor(var_correction_294, 0.001)
        var_correction_294 = None
        sqrt_default_147 = torch.ops.aten.sqrt.default(add_tensor_649)
        add_tensor_649 = None
        reciprocal_default_147 = torch.ops.aten.reciprocal.default(sqrt_default_147)
        sqrt_default_147 = None
        sub_tensor_147 = torch.ops.aten.sub.Tensor(
            convolution_default_272, mean_dim_147
        )
        mul_tensor_1029 = torch.ops.aten.mul.Tensor(
            sub_tensor_147, reciprocal_default_147
        )
        sub_tensor_147 = None
        squeeze_dim_882 = torch.ops.aten.squeeze.dim(mean_dim_147, 3)
        mean_dim_147 = None
        squeeze_dim_883 = torch.ops.aten.squeeze.dim(squeeze_dim_882, 2)
        squeeze_dim_882 = None
        squeeze_dim_884 = torch.ops.aten.squeeze.dim(squeeze_dim_883, 0)
        squeeze_dim_883 = None
        squeeze_dim_885 = torch.ops.aten.squeeze.dim(reciprocal_default_147, 3)
        reciprocal_default_147 = None
        squeeze_dim_886 = torch.ops.aten.squeeze.dim(squeeze_dim_885, 2)
        squeeze_dim_885 = None
        squeeze_dim_887 = torch.ops.aten.squeeze.dim(squeeze_dim_886, 0)
        squeeze_dim_886 = None
        unsqueeze_default_588 = torch.ops.aten.unsqueeze.default(primals_570, -1)
        unsqueeze_default_589 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_588, -1
        )
        unsqueeze_default_588 = None
        unsqueeze_default_590 = torch.ops.aten.unsqueeze.default(primals_571, -1)
        primals_571 = None
        unsqueeze_default_591 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_590, -1
        )
        unsqueeze_default_590 = None
        mul_tensor_1035 = torch.ops.aten.mul.Tensor(
            mul_tensor_1029, unsqueeze_default_589
        )
        mul_tensor_1029 = unsqueeze_default_589 = None
        add_tensor_652 = torch.ops.aten.add.Tensor(
            mul_tensor_1035, unsqueeze_default_591
        )
        mul_tensor_1035 = unsqueeze_default_591 = None
        relu_default_147 = torch.ops.aten.relu.default(add_tensor_640)
        convolution_default_273 = torch.ops.aten.convolution.default(
            relu_default_147,
            primals_572,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_274 = torch.ops.aten.convolution.default(
            convolution_default_273,
            primals_573,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_296 = torch.ops.aten.var.correction(
            convolution_default_274, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_148 = torch.ops.aten.mean.dim(convolution_default_274, [0, 2, 3], True)
        add_tensor_653 = torch.ops.aten.add.Tensor(var_correction_296, 0.001)
        var_correction_296 = None
        sqrt_default_148 = torch.ops.aten.sqrt.default(add_tensor_653)
        add_tensor_653 = None
        reciprocal_default_148 = torch.ops.aten.reciprocal.default(sqrt_default_148)
        sqrt_default_148 = None
        sub_tensor_148 = torch.ops.aten.sub.Tensor(
            convolution_default_274, mean_dim_148
        )
        mul_tensor_1036 = torch.ops.aten.mul.Tensor(
            sub_tensor_148, reciprocal_default_148
        )
        sub_tensor_148 = None
        squeeze_dim_888 = torch.ops.aten.squeeze.dim(mean_dim_148, 3)
        mean_dim_148 = None
        squeeze_dim_889 = torch.ops.aten.squeeze.dim(squeeze_dim_888, 2)
        squeeze_dim_888 = None
        squeeze_dim_890 = torch.ops.aten.squeeze.dim(squeeze_dim_889, 0)
        squeeze_dim_889 = None
        squeeze_dim_891 = torch.ops.aten.squeeze.dim(reciprocal_default_148, 3)
        reciprocal_default_148 = None
        squeeze_dim_892 = torch.ops.aten.squeeze.dim(squeeze_dim_891, 2)
        squeeze_dim_891 = None
        squeeze_dim_893 = torch.ops.aten.squeeze.dim(squeeze_dim_892, 0)
        squeeze_dim_892 = None
        unsqueeze_default_592 = torch.ops.aten.unsqueeze.default(primals_574, -1)
        unsqueeze_default_593 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_592, -1
        )
        unsqueeze_default_592 = None
        unsqueeze_default_594 = torch.ops.aten.unsqueeze.default(primals_575, -1)
        primals_575 = None
        unsqueeze_default_595 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_594, -1
        )
        unsqueeze_default_594 = None
        mul_tensor_1042 = torch.ops.aten.mul.Tensor(
            mul_tensor_1036, unsqueeze_default_593
        )
        mul_tensor_1036 = unsqueeze_default_593 = None
        add_tensor_656 = torch.ops.aten.add.Tensor(
            mul_tensor_1042, unsqueeze_default_595
        )
        mul_tensor_1042 = unsqueeze_default_595 = None
        relu_default_148 = torch.ops.aten.relu.default(add_tensor_656)
        add_tensor_656 = None
        convolution_default_275 = torch.ops.aten.convolution.default(
            relu_default_148,
            primals_576,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_276 = torch.ops.aten.convolution.default(
            convolution_default_275,
            primals_577,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_298 = torch.ops.aten.var.correction(
            convolution_default_276, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_149 = torch.ops.aten.mean.dim(convolution_default_276, [0, 2, 3], True)
        add_tensor_657 = torch.ops.aten.add.Tensor(var_correction_298, 0.001)
        var_correction_298 = None
        sqrt_default_149 = torch.ops.aten.sqrt.default(add_tensor_657)
        add_tensor_657 = None
        reciprocal_default_149 = torch.ops.aten.reciprocal.default(sqrt_default_149)
        sqrt_default_149 = None
        sub_tensor_149 = torch.ops.aten.sub.Tensor(
            convolution_default_276, mean_dim_149
        )
        mul_tensor_1043 = torch.ops.aten.mul.Tensor(
            sub_tensor_149, reciprocal_default_149
        )
        sub_tensor_149 = None
        squeeze_dim_894 = torch.ops.aten.squeeze.dim(mean_dim_149, 3)
        mean_dim_149 = None
        squeeze_dim_895 = torch.ops.aten.squeeze.dim(squeeze_dim_894, 2)
        squeeze_dim_894 = None
        squeeze_dim_896 = torch.ops.aten.squeeze.dim(squeeze_dim_895, 0)
        squeeze_dim_895 = None
        squeeze_dim_897 = torch.ops.aten.squeeze.dim(reciprocal_default_149, 3)
        reciprocal_default_149 = None
        squeeze_dim_898 = torch.ops.aten.squeeze.dim(squeeze_dim_897, 2)
        squeeze_dim_897 = None
        squeeze_dim_899 = torch.ops.aten.squeeze.dim(squeeze_dim_898, 0)
        squeeze_dim_898 = None
        unsqueeze_default_596 = torch.ops.aten.unsqueeze.default(primals_578, -1)
        unsqueeze_default_597 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_596, -1
        )
        unsqueeze_default_596 = None
        unsqueeze_default_598 = torch.ops.aten.unsqueeze.default(primals_579, -1)
        primals_579 = None
        unsqueeze_default_599 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_598, -1
        )
        unsqueeze_default_598 = None
        mul_tensor_1049 = torch.ops.aten.mul.Tensor(
            mul_tensor_1043, unsqueeze_default_597
        )
        mul_tensor_1043 = unsqueeze_default_597 = None
        add_tensor_660 = torch.ops.aten.add.Tensor(
            mul_tensor_1049, unsqueeze_default_599
        )
        mul_tensor_1049 = unsqueeze_default_599 = None
        add_tensor_661 = torch.ops.aten.add.Tensor(add_tensor_652, add_tensor_660)
        add_tensor_652 = add_tensor_660 = None
        convolution_default_277 = torch.ops.aten.convolution.default(
            relu_default_147,
            primals_580,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_278 = torch.ops.aten.convolution.default(
            convolution_default_277,
            primals_581,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_300 = torch.ops.aten.var.correction(
            convolution_default_278, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_150 = torch.ops.aten.mean.dim(convolution_default_278, [0, 2, 3], True)
        add_tensor_662 = torch.ops.aten.add.Tensor(var_correction_300, 0.001)
        var_correction_300 = None
        sqrt_default_150 = torch.ops.aten.sqrt.default(add_tensor_662)
        add_tensor_662 = None
        reciprocal_default_150 = torch.ops.aten.reciprocal.default(sqrt_default_150)
        sqrt_default_150 = None
        sub_tensor_150 = torch.ops.aten.sub.Tensor(
            convolution_default_278, mean_dim_150
        )
        mul_tensor_1050 = torch.ops.aten.mul.Tensor(
            sub_tensor_150, reciprocal_default_150
        )
        sub_tensor_150 = None
        squeeze_dim_900 = torch.ops.aten.squeeze.dim(mean_dim_150, 3)
        mean_dim_150 = None
        squeeze_dim_901 = torch.ops.aten.squeeze.dim(squeeze_dim_900, 2)
        squeeze_dim_900 = None
        squeeze_dim_902 = torch.ops.aten.squeeze.dim(squeeze_dim_901, 0)
        squeeze_dim_901 = None
        squeeze_dim_903 = torch.ops.aten.squeeze.dim(reciprocal_default_150, 3)
        reciprocal_default_150 = None
        squeeze_dim_904 = torch.ops.aten.squeeze.dim(squeeze_dim_903, 2)
        squeeze_dim_903 = None
        squeeze_dim_905 = torch.ops.aten.squeeze.dim(squeeze_dim_904, 0)
        squeeze_dim_904 = None
        unsqueeze_default_600 = torch.ops.aten.unsqueeze.default(primals_582, -1)
        unsqueeze_default_601 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_600, -1
        )
        unsqueeze_default_600 = None
        unsqueeze_default_602 = torch.ops.aten.unsqueeze.default(primals_583, -1)
        primals_583 = None
        unsqueeze_default_603 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_602, -1
        )
        unsqueeze_default_602 = None
        mul_tensor_1056 = torch.ops.aten.mul.Tensor(
            mul_tensor_1050, unsqueeze_default_601
        )
        mul_tensor_1050 = unsqueeze_default_601 = None
        add_tensor_665 = torch.ops.aten.add.Tensor(
            mul_tensor_1056, unsqueeze_default_603
        )
        mul_tensor_1056 = unsqueeze_default_603 = None
        relu_default_150 = torch.ops.aten.relu.default(add_tensor_665)
        add_tensor_665 = None
        convolution_default_279 = torch.ops.aten.convolution.default(
            relu_default_150,
            primals_584,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_280 = torch.ops.aten.convolution.default(
            convolution_default_279,
            primals_585,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_302 = torch.ops.aten.var.correction(
            convolution_default_280, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_151 = torch.ops.aten.mean.dim(convolution_default_280, [0, 2, 3], True)
        add_tensor_666 = torch.ops.aten.add.Tensor(var_correction_302, 0.001)
        var_correction_302 = None
        sqrt_default_151 = torch.ops.aten.sqrt.default(add_tensor_666)
        add_tensor_666 = None
        reciprocal_default_151 = torch.ops.aten.reciprocal.default(sqrt_default_151)
        sqrt_default_151 = None
        sub_tensor_151 = torch.ops.aten.sub.Tensor(
            convolution_default_280, mean_dim_151
        )
        mul_tensor_1057 = torch.ops.aten.mul.Tensor(
            sub_tensor_151, reciprocal_default_151
        )
        sub_tensor_151 = None
        squeeze_dim_906 = torch.ops.aten.squeeze.dim(mean_dim_151, 3)
        mean_dim_151 = None
        squeeze_dim_907 = torch.ops.aten.squeeze.dim(squeeze_dim_906, 2)
        squeeze_dim_906 = None
        squeeze_dim_908 = torch.ops.aten.squeeze.dim(squeeze_dim_907, 0)
        squeeze_dim_907 = None
        squeeze_dim_909 = torch.ops.aten.squeeze.dim(reciprocal_default_151, 3)
        reciprocal_default_151 = None
        squeeze_dim_910 = torch.ops.aten.squeeze.dim(squeeze_dim_909, 2)
        squeeze_dim_909 = None
        squeeze_dim_911 = torch.ops.aten.squeeze.dim(squeeze_dim_910, 0)
        squeeze_dim_910 = None
        unsqueeze_default_604 = torch.ops.aten.unsqueeze.default(primals_586, -1)
        unsqueeze_default_605 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_604, -1
        )
        unsqueeze_default_604 = None
        unsqueeze_default_606 = torch.ops.aten.unsqueeze.default(primals_587, -1)
        primals_587 = None
        unsqueeze_default_607 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_606, -1
        )
        unsqueeze_default_606 = None
        mul_tensor_1063 = torch.ops.aten.mul.Tensor(
            mul_tensor_1057, unsqueeze_default_605
        )
        mul_tensor_1057 = unsqueeze_default_605 = None
        add_tensor_669 = torch.ops.aten.add.Tensor(
            mul_tensor_1063, unsqueeze_default_607
        )
        mul_tensor_1063 = unsqueeze_default_607 = None
        convolution_default_281 = torch.ops.aten.convolution.default(
            relu_default_147,
            primals_588,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_147 = None
        convolution_default_282 = torch.ops.aten.convolution.default(
            convolution_default_281,
            primals_589,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_304 = torch.ops.aten.var.correction(
            convolution_default_282, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_152 = torch.ops.aten.mean.dim(convolution_default_282, [0, 2, 3], True)
        add_tensor_670 = torch.ops.aten.add.Tensor(var_correction_304, 0.001)
        var_correction_304 = None
        sqrt_default_152 = torch.ops.aten.sqrt.default(add_tensor_670)
        add_tensor_670 = None
        reciprocal_default_152 = torch.ops.aten.reciprocal.default(sqrt_default_152)
        sqrt_default_152 = None
        sub_tensor_152 = torch.ops.aten.sub.Tensor(
            convolution_default_282, mean_dim_152
        )
        mul_tensor_1064 = torch.ops.aten.mul.Tensor(
            sub_tensor_152, reciprocal_default_152
        )
        sub_tensor_152 = None
        squeeze_dim_912 = torch.ops.aten.squeeze.dim(mean_dim_152, 3)
        mean_dim_152 = None
        squeeze_dim_913 = torch.ops.aten.squeeze.dim(squeeze_dim_912, 2)
        squeeze_dim_912 = None
        squeeze_dim_914 = torch.ops.aten.squeeze.dim(squeeze_dim_913, 0)
        squeeze_dim_913 = None
        squeeze_dim_915 = torch.ops.aten.squeeze.dim(reciprocal_default_152, 3)
        reciprocal_default_152 = None
        squeeze_dim_916 = torch.ops.aten.squeeze.dim(squeeze_dim_915, 2)
        squeeze_dim_915 = None
        squeeze_dim_917 = torch.ops.aten.squeeze.dim(squeeze_dim_916, 0)
        squeeze_dim_916 = None
        unsqueeze_default_608 = torch.ops.aten.unsqueeze.default(primals_590, -1)
        unsqueeze_default_609 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_608, -1
        )
        unsqueeze_default_608 = None
        unsqueeze_default_610 = torch.ops.aten.unsqueeze.default(primals_591, -1)
        primals_591 = None
        unsqueeze_default_611 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_610, -1
        )
        unsqueeze_default_610 = None
        mul_tensor_1070 = torch.ops.aten.mul.Tensor(
            mul_tensor_1064, unsqueeze_default_609
        )
        mul_tensor_1064 = unsqueeze_default_609 = None
        add_tensor_673 = torch.ops.aten.add.Tensor(
            mul_tensor_1070, unsqueeze_default_611
        )
        mul_tensor_1070 = unsqueeze_default_611 = None
        relu_default_152 = torch.ops.aten.relu.default(add_tensor_673)
        add_tensor_673 = None
        convolution_default_283 = torch.ops.aten.convolution.default(
            relu_default_152,
            primals_592,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_284 = torch.ops.aten.convolution.default(
            convolution_default_283,
            primals_593,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_306 = torch.ops.aten.var.correction(
            convolution_default_284, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_153 = torch.ops.aten.mean.dim(convolution_default_284, [0, 2, 3], True)
        add_tensor_674 = torch.ops.aten.add.Tensor(var_correction_306, 0.001)
        var_correction_306 = None
        sqrt_default_153 = torch.ops.aten.sqrt.default(add_tensor_674)
        add_tensor_674 = None
        reciprocal_default_153 = torch.ops.aten.reciprocal.default(sqrt_default_153)
        sqrt_default_153 = None
        sub_tensor_153 = torch.ops.aten.sub.Tensor(
            convolution_default_284, mean_dim_153
        )
        mul_tensor_1071 = torch.ops.aten.mul.Tensor(
            sub_tensor_153, reciprocal_default_153
        )
        sub_tensor_153 = None
        squeeze_dim_918 = torch.ops.aten.squeeze.dim(mean_dim_153, 3)
        mean_dim_153 = None
        squeeze_dim_919 = torch.ops.aten.squeeze.dim(squeeze_dim_918, 2)
        squeeze_dim_918 = None
        squeeze_dim_920 = torch.ops.aten.squeeze.dim(squeeze_dim_919, 0)
        squeeze_dim_919 = None
        squeeze_dim_921 = torch.ops.aten.squeeze.dim(reciprocal_default_153, 3)
        reciprocal_default_153 = None
        squeeze_dim_922 = torch.ops.aten.squeeze.dim(squeeze_dim_921, 2)
        squeeze_dim_921 = None
        squeeze_dim_923 = torch.ops.aten.squeeze.dim(squeeze_dim_922, 0)
        squeeze_dim_922 = None
        unsqueeze_default_612 = torch.ops.aten.unsqueeze.default(primals_594, -1)
        unsqueeze_default_613 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_612, -1
        )
        unsqueeze_default_612 = None
        unsqueeze_default_614 = torch.ops.aten.unsqueeze.default(primals_595, -1)
        primals_595 = None
        unsqueeze_default_615 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_614, -1
        )
        unsqueeze_default_614 = None
        mul_tensor_1077 = torch.ops.aten.mul.Tensor(
            mul_tensor_1071, unsqueeze_default_613
        )
        mul_tensor_1071 = unsqueeze_default_613 = None
        add_tensor_677 = torch.ops.aten.add.Tensor(
            mul_tensor_1077, unsqueeze_default_615
        )
        mul_tensor_1077 = unsqueeze_default_615 = None
        add_tensor_678 = torch.ops.aten.add.Tensor(add_tensor_669, add_tensor_677)
        add_tensor_669 = add_tensor_677 = None
        avg_pool2d_default_39 = torch.ops.aten.avg_pool2d.default(
            add_tensor_644, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_679 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_39, add_tensor_640
        )
        avg_pool2d_default_39 = None
        avg_pool2d_default_40 = torch.ops.aten.avg_pool2d.default(
            add_tensor_640, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_680 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_40, avg_pool2d_default_40
        )
        avg_pool2d_default_40 = None
        convolution_default_285 = torch.ops.aten.convolution.default(
            relu_default_145,
            primals_596,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_145 = None
        convolution_default_286 = torch.ops.aten.convolution.default(
            convolution_default_285,
            primals_597,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_308 = torch.ops.aten.var.correction(
            convolution_default_286, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_154 = torch.ops.aten.mean.dim(convolution_default_286, [0, 2, 3], True)
        add_tensor_681 = torch.ops.aten.add.Tensor(var_correction_308, 0.001)
        var_correction_308 = None
        sqrt_default_154 = torch.ops.aten.sqrt.default(add_tensor_681)
        add_tensor_681 = None
        reciprocal_default_154 = torch.ops.aten.reciprocal.default(sqrt_default_154)
        sqrt_default_154 = None
        sub_tensor_154 = torch.ops.aten.sub.Tensor(
            convolution_default_286, mean_dim_154
        )
        mul_tensor_1078 = torch.ops.aten.mul.Tensor(
            sub_tensor_154, reciprocal_default_154
        )
        sub_tensor_154 = None
        squeeze_dim_924 = torch.ops.aten.squeeze.dim(mean_dim_154, 3)
        mean_dim_154 = None
        squeeze_dim_925 = torch.ops.aten.squeeze.dim(squeeze_dim_924, 2)
        squeeze_dim_924 = None
        squeeze_dim_926 = torch.ops.aten.squeeze.dim(squeeze_dim_925, 0)
        squeeze_dim_925 = None
        squeeze_dim_927 = torch.ops.aten.squeeze.dim(reciprocal_default_154, 3)
        reciprocal_default_154 = None
        squeeze_dim_928 = torch.ops.aten.squeeze.dim(squeeze_dim_927, 2)
        squeeze_dim_927 = None
        squeeze_dim_929 = torch.ops.aten.squeeze.dim(squeeze_dim_928, 0)
        squeeze_dim_928 = None
        unsqueeze_default_616 = torch.ops.aten.unsqueeze.default(primals_598, -1)
        unsqueeze_default_617 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_616, -1
        )
        unsqueeze_default_616 = None
        unsqueeze_default_618 = torch.ops.aten.unsqueeze.default(primals_599, -1)
        primals_599 = None
        unsqueeze_default_619 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_618, -1
        )
        unsqueeze_default_618 = None
        mul_tensor_1084 = torch.ops.aten.mul.Tensor(
            mul_tensor_1078, unsqueeze_default_617
        )
        mul_tensor_1078 = unsqueeze_default_617 = None
        add_tensor_684 = torch.ops.aten.add.Tensor(
            mul_tensor_1084, unsqueeze_default_619
        )
        mul_tensor_1084 = unsqueeze_default_619 = None
        relu_default_154 = torch.ops.aten.relu.default(add_tensor_684)
        add_tensor_684 = None
        convolution_default_287 = torch.ops.aten.convolution.default(
            relu_default_154,
            primals_600,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_288 = torch.ops.aten.convolution.default(
            convolution_default_287,
            primals_601,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_310 = torch.ops.aten.var.correction(
            convolution_default_288, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_155 = torch.ops.aten.mean.dim(convolution_default_288, [0, 2, 3], True)
        add_tensor_685 = torch.ops.aten.add.Tensor(var_correction_310, 0.001)
        var_correction_310 = None
        sqrt_default_155 = torch.ops.aten.sqrt.default(add_tensor_685)
        add_tensor_685 = None
        reciprocal_default_155 = torch.ops.aten.reciprocal.default(sqrt_default_155)
        sqrt_default_155 = None
        sub_tensor_155 = torch.ops.aten.sub.Tensor(
            convolution_default_288, mean_dim_155
        )
        mul_tensor_1085 = torch.ops.aten.mul.Tensor(
            sub_tensor_155, reciprocal_default_155
        )
        sub_tensor_155 = None
        squeeze_dim_930 = torch.ops.aten.squeeze.dim(mean_dim_155, 3)
        mean_dim_155 = None
        squeeze_dim_931 = torch.ops.aten.squeeze.dim(squeeze_dim_930, 2)
        squeeze_dim_930 = None
        squeeze_dim_932 = torch.ops.aten.squeeze.dim(squeeze_dim_931, 0)
        squeeze_dim_931 = None
        squeeze_dim_933 = torch.ops.aten.squeeze.dim(reciprocal_default_155, 3)
        reciprocal_default_155 = None
        squeeze_dim_934 = torch.ops.aten.squeeze.dim(squeeze_dim_933, 2)
        squeeze_dim_933 = None
        squeeze_dim_935 = torch.ops.aten.squeeze.dim(squeeze_dim_934, 0)
        squeeze_dim_934 = None
        unsqueeze_default_620 = torch.ops.aten.unsqueeze.default(primals_602, -1)
        unsqueeze_default_621 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_620, -1
        )
        unsqueeze_default_620 = None
        unsqueeze_default_622 = torch.ops.aten.unsqueeze.default(primals_603, -1)
        primals_603 = None
        unsqueeze_default_623 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_622, -1
        )
        unsqueeze_default_622 = None
        mul_tensor_1091 = torch.ops.aten.mul.Tensor(
            mul_tensor_1085, unsqueeze_default_621
        )
        mul_tensor_1085 = unsqueeze_default_621 = None
        add_tensor_688 = torch.ops.aten.add.Tensor(
            mul_tensor_1091, unsqueeze_default_623
        )
        mul_tensor_1091 = unsqueeze_default_623 = None
        add_tensor_689 = torch.ops.aten.add.Tensor(add_tensor_688, add_tensor_644)
        add_tensor_688 = add_tensor_644 = None
        cat_default_15 = torch.ops.aten.cat.default(
            [
                add_tensor_640,
                add_tensor_661,
                add_tensor_678,
                add_tensor_679,
                add_tensor_680,
                add_tensor_689,
            ],
            1,
        )
        add_tensor_640 = (
            add_tensor_661
        ) = add_tensor_678 = add_tensor_679 = add_tensor_680 = add_tensor_689 = None
        convolution_default_289 = torch.ops.aten.convolution.default(
            relu_default_144,
            primals_604,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_312 = torch.ops.aten.var.correction(
            convolution_default_289, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_156 = torch.ops.aten.mean.dim(convolution_default_289, [0, 2, 3], True)
        add_tensor_690 = torch.ops.aten.add.Tensor(var_correction_312, 0.001)
        var_correction_312 = None
        sqrt_default_156 = torch.ops.aten.sqrt.default(add_tensor_690)
        add_tensor_690 = None
        reciprocal_default_156 = torch.ops.aten.reciprocal.default(sqrt_default_156)
        sqrt_default_156 = None
        sub_tensor_156 = torch.ops.aten.sub.Tensor(
            convolution_default_289, mean_dim_156
        )
        mul_tensor_1092 = torch.ops.aten.mul.Tensor(
            sub_tensor_156, reciprocal_default_156
        )
        sub_tensor_156 = None
        unsqueeze_default_624 = torch.ops.aten.unsqueeze.default(primals_605, -1)
        unsqueeze_default_625 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_624, -1
        )
        unsqueeze_default_624 = None
        unsqueeze_default_626 = torch.ops.aten.unsqueeze.default(primals_606, -1)
        unsqueeze_default_627 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_626, -1
        )
        unsqueeze_default_626 = None
        mul_tensor_1098 = torch.ops.aten.mul.Tensor(
            mul_tensor_1092, unsqueeze_default_625
        )
        mul_tensor_1092 = unsqueeze_default_625 = None
        add_tensor_693 = torch.ops.aten.add.Tensor(
            mul_tensor_1098, unsqueeze_default_627
        )
        mul_tensor_1098 = unsqueeze_default_627 = None
        relu_default_156 = torch.ops.aten.relu.default(cat_default_15)
        cat_default_15 = None
        convolution_default_290 = torch.ops.aten.convolution.default(
            relu_default_156,
            primals_607,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_314 = torch.ops.aten.var.correction(
            convolution_default_290, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_157 = torch.ops.aten.mean.dim(convolution_default_290, [0, 2, 3], True)
        add_tensor_694 = torch.ops.aten.add.Tensor(var_correction_314, 0.001)
        var_correction_314 = None
        sqrt_default_157 = torch.ops.aten.sqrt.default(add_tensor_694)
        add_tensor_694 = None
        reciprocal_default_157 = torch.ops.aten.reciprocal.default(sqrt_default_157)
        sqrt_default_157 = None
        sub_tensor_157 = torch.ops.aten.sub.Tensor(
            convolution_default_290, mean_dim_157
        )
        mul_tensor_1099 = torch.ops.aten.mul.Tensor(
            sub_tensor_157, reciprocal_default_157
        )
        sub_tensor_157 = None
        unsqueeze_default_628 = torch.ops.aten.unsqueeze.default(primals_608, -1)
        unsqueeze_default_629 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_628, -1
        )
        unsqueeze_default_628 = None
        unsqueeze_default_630 = torch.ops.aten.unsqueeze.default(primals_609, -1)
        unsqueeze_default_631 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_630, -1
        )
        unsqueeze_default_630 = None
        mul_tensor_1105 = torch.ops.aten.mul.Tensor(
            mul_tensor_1099, unsqueeze_default_629
        )
        mul_tensor_1099 = unsqueeze_default_629 = None
        add_tensor_697 = torch.ops.aten.add.Tensor(
            mul_tensor_1105, unsqueeze_default_631
        )
        mul_tensor_1105 = unsqueeze_default_631 = None
        relu_default_157 = torch.ops.aten.relu.default(add_tensor_697)
        convolution_default_291 = torch.ops.aten.convolution.default(
            relu_default_157,
            primals_610,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_292 = torch.ops.aten.convolution.default(
            convolution_default_291,
            primals_611,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_316 = torch.ops.aten.var.correction(
            convolution_default_292, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_158 = torch.ops.aten.mean.dim(convolution_default_292, [0, 2, 3], True)
        add_tensor_698 = torch.ops.aten.add.Tensor(var_correction_316, 0.001)
        var_correction_316 = None
        sqrt_default_158 = torch.ops.aten.sqrt.default(add_tensor_698)
        add_tensor_698 = None
        reciprocal_default_158 = torch.ops.aten.reciprocal.default(sqrt_default_158)
        sqrt_default_158 = None
        sub_tensor_158 = torch.ops.aten.sub.Tensor(
            convolution_default_292, mean_dim_158
        )
        mul_tensor_1106 = torch.ops.aten.mul.Tensor(
            sub_tensor_158, reciprocal_default_158
        )
        sub_tensor_158 = None
        squeeze_dim_948 = torch.ops.aten.squeeze.dim(mean_dim_158, 3)
        mean_dim_158 = None
        squeeze_dim_949 = torch.ops.aten.squeeze.dim(squeeze_dim_948, 2)
        squeeze_dim_948 = None
        squeeze_dim_950 = torch.ops.aten.squeeze.dim(squeeze_dim_949, 0)
        squeeze_dim_949 = None
        squeeze_dim_951 = torch.ops.aten.squeeze.dim(reciprocal_default_158, 3)
        reciprocal_default_158 = None
        squeeze_dim_952 = torch.ops.aten.squeeze.dim(squeeze_dim_951, 2)
        squeeze_dim_951 = None
        squeeze_dim_953 = torch.ops.aten.squeeze.dim(squeeze_dim_952, 0)
        squeeze_dim_952 = None
        unsqueeze_default_632 = torch.ops.aten.unsqueeze.default(primals_612, -1)
        unsqueeze_default_633 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_632, -1
        )
        unsqueeze_default_632 = None
        unsqueeze_default_634 = torch.ops.aten.unsqueeze.default(primals_613, -1)
        primals_613 = None
        unsqueeze_default_635 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_634, -1
        )
        unsqueeze_default_634 = None
        mul_tensor_1112 = torch.ops.aten.mul.Tensor(
            mul_tensor_1106, unsqueeze_default_633
        )
        mul_tensor_1106 = unsqueeze_default_633 = None
        add_tensor_701 = torch.ops.aten.add.Tensor(
            mul_tensor_1112, unsqueeze_default_635
        )
        mul_tensor_1112 = unsqueeze_default_635 = None
        relu_default_158 = torch.ops.aten.relu.default(add_tensor_701)
        add_tensor_701 = None
        convolution_default_293 = torch.ops.aten.convolution.default(
            relu_default_158,
            primals_614,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_294 = torch.ops.aten.convolution.default(
            convolution_default_293,
            primals_615,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_318 = torch.ops.aten.var.correction(
            convolution_default_294, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_159 = torch.ops.aten.mean.dim(convolution_default_294, [0, 2, 3], True)
        add_tensor_702 = torch.ops.aten.add.Tensor(var_correction_318, 0.001)
        var_correction_318 = None
        sqrt_default_159 = torch.ops.aten.sqrt.default(add_tensor_702)
        add_tensor_702 = None
        reciprocal_default_159 = torch.ops.aten.reciprocal.default(sqrt_default_159)
        sqrt_default_159 = None
        sub_tensor_159 = torch.ops.aten.sub.Tensor(
            convolution_default_294, mean_dim_159
        )
        mul_tensor_1113 = torch.ops.aten.mul.Tensor(
            sub_tensor_159, reciprocal_default_159
        )
        sub_tensor_159 = None
        squeeze_dim_954 = torch.ops.aten.squeeze.dim(mean_dim_159, 3)
        mean_dim_159 = None
        squeeze_dim_955 = torch.ops.aten.squeeze.dim(squeeze_dim_954, 2)
        squeeze_dim_954 = None
        squeeze_dim_956 = torch.ops.aten.squeeze.dim(squeeze_dim_955, 0)
        squeeze_dim_955 = None
        squeeze_dim_957 = torch.ops.aten.squeeze.dim(reciprocal_default_159, 3)
        reciprocal_default_159 = None
        squeeze_dim_958 = torch.ops.aten.squeeze.dim(squeeze_dim_957, 2)
        squeeze_dim_957 = None
        squeeze_dim_959 = torch.ops.aten.squeeze.dim(squeeze_dim_958, 0)
        squeeze_dim_958 = None
        unsqueeze_default_636 = torch.ops.aten.unsqueeze.default(primals_616, -1)
        unsqueeze_default_637 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_636, -1
        )
        unsqueeze_default_636 = None
        unsqueeze_default_638 = torch.ops.aten.unsqueeze.default(primals_617, -1)
        primals_617 = None
        unsqueeze_default_639 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_638, -1
        )
        unsqueeze_default_638 = None
        mul_tensor_1119 = torch.ops.aten.mul.Tensor(
            mul_tensor_1113, unsqueeze_default_637
        )
        mul_tensor_1113 = unsqueeze_default_637 = None
        add_tensor_705 = torch.ops.aten.add.Tensor(
            mul_tensor_1119, unsqueeze_default_639
        )
        mul_tensor_1119 = unsqueeze_default_639 = None
        relu_default_159 = torch.ops.aten.relu.default(add_tensor_693)
        convolution_default_295 = torch.ops.aten.convolution.default(
            relu_default_159,
            primals_618,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_296 = torch.ops.aten.convolution.default(
            convolution_default_295,
            primals_619,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_320 = torch.ops.aten.var.correction(
            convolution_default_296, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_160 = torch.ops.aten.mean.dim(convolution_default_296, [0, 2, 3], True)
        add_tensor_706 = torch.ops.aten.add.Tensor(var_correction_320, 0.001)
        var_correction_320 = None
        sqrt_default_160 = torch.ops.aten.sqrt.default(add_tensor_706)
        add_tensor_706 = None
        reciprocal_default_160 = torch.ops.aten.reciprocal.default(sqrt_default_160)
        sqrt_default_160 = None
        sub_tensor_160 = torch.ops.aten.sub.Tensor(
            convolution_default_296, mean_dim_160
        )
        mul_tensor_1120 = torch.ops.aten.mul.Tensor(
            sub_tensor_160, reciprocal_default_160
        )
        sub_tensor_160 = None
        squeeze_dim_960 = torch.ops.aten.squeeze.dim(mean_dim_160, 3)
        mean_dim_160 = None
        squeeze_dim_961 = torch.ops.aten.squeeze.dim(squeeze_dim_960, 2)
        squeeze_dim_960 = None
        squeeze_dim_962 = torch.ops.aten.squeeze.dim(squeeze_dim_961, 0)
        squeeze_dim_961 = None
        squeeze_dim_963 = torch.ops.aten.squeeze.dim(reciprocal_default_160, 3)
        reciprocal_default_160 = None
        squeeze_dim_964 = torch.ops.aten.squeeze.dim(squeeze_dim_963, 2)
        squeeze_dim_963 = None
        squeeze_dim_965 = torch.ops.aten.squeeze.dim(squeeze_dim_964, 0)
        squeeze_dim_964 = None
        unsqueeze_default_640 = torch.ops.aten.unsqueeze.default(primals_620, -1)
        unsqueeze_default_641 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_640, -1
        )
        unsqueeze_default_640 = None
        unsqueeze_default_642 = torch.ops.aten.unsqueeze.default(primals_621, -1)
        primals_621 = None
        unsqueeze_default_643 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_642, -1
        )
        unsqueeze_default_642 = None
        mul_tensor_1126 = torch.ops.aten.mul.Tensor(
            mul_tensor_1120, unsqueeze_default_641
        )
        mul_tensor_1120 = unsqueeze_default_641 = None
        add_tensor_709 = torch.ops.aten.add.Tensor(
            mul_tensor_1126, unsqueeze_default_643
        )
        mul_tensor_1126 = unsqueeze_default_643 = None
        relu_default_160 = torch.ops.aten.relu.default(add_tensor_709)
        add_tensor_709 = None
        convolution_default_297 = torch.ops.aten.convolution.default(
            relu_default_160,
            primals_622,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_298 = torch.ops.aten.convolution.default(
            convolution_default_297,
            primals_623,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_322 = torch.ops.aten.var.correction(
            convolution_default_298, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_161 = torch.ops.aten.mean.dim(convolution_default_298, [0, 2, 3], True)
        add_tensor_710 = torch.ops.aten.add.Tensor(var_correction_322, 0.001)
        var_correction_322 = None
        sqrt_default_161 = torch.ops.aten.sqrt.default(add_tensor_710)
        add_tensor_710 = None
        reciprocal_default_161 = torch.ops.aten.reciprocal.default(sqrt_default_161)
        sqrt_default_161 = None
        sub_tensor_161 = torch.ops.aten.sub.Tensor(
            convolution_default_298, mean_dim_161
        )
        mul_tensor_1127 = torch.ops.aten.mul.Tensor(
            sub_tensor_161, reciprocal_default_161
        )
        sub_tensor_161 = None
        squeeze_dim_966 = torch.ops.aten.squeeze.dim(mean_dim_161, 3)
        mean_dim_161 = None
        squeeze_dim_967 = torch.ops.aten.squeeze.dim(squeeze_dim_966, 2)
        squeeze_dim_966 = None
        squeeze_dim_968 = torch.ops.aten.squeeze.dim(squeeze_dim_967, 0)
        squeeze_dim_967 = None
        squeeze_dim_969 = torch.ops.aten.squeeze.dim(reciprocal_default_161, 3)
        reciprocal_default_161 = None
        squeeze_dim_970 = torch.ops.aten.squeeze.dim(squeeze_dim_969, 2)
        squeeze_dim_969 = None
        squeeze_dim_971 = torch.ops.aten.squeeze.dim(squeeze_dim_970, 0)
        squeeze_dim_970 = None
        unsqueeze_default_644 = torch.ops.aten.unsqueeze.default(primals_624, -1)
        unsqueeze_default_645 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_644, -1
        )
        unsqueeze_default_644 = None
        unsqueeze_default_646 = torch.ops.aten.unsqueeze.default(primals_625, -1)
        primals_625 = None
        unsqueeze_default_647 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_646, -1
        )
        unsqueeze_default_646 = None
        mul_tensor_1133 = torch.ops.aten.mul.Tensor(
            mul_tensor_1127, unsqueeze_default_645
        )
        mul_tensor_1127 = unsqueeze_default_645 = None
        add_tensor_713 = torch.ops.aten.add.Tensor(
            mul_tensor_1133, unsqueeze_default_647
        )
        mul_tensor_1133 = unsqueeze_default_647 = None
        add_tensor_714 = torch.ops.aten.add.Tensor(add_tensor_705, add_tensor_713)
        add_tensor_705 = add_tensor_713 = None
        convolution_default_299 = torch.ops.aten.convolution.default(
            relu_default_159,
            primals_626,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_300 = torch.ops.aten.convolution.default(
            convolution_default_299,
            primals_627,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_324 = torch.ops.aten.var.correction(
            convolution_default_300, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_162 = torch.ops.aten.mean.dim(convolution_default_300, [0, 2, 3], True)
        add_tensor_715 = torch.ops.aten.add.Tensor(var_correction_324, 0.001)
        var_correction_324 = None
        sqrt_default_162 = torch.ops.aten.sqrt.default(add_tensor_715)
        add_tensor_715 = None
        reciprocal_default_162 = torch.ops.aten.reciprocal.default(sqrt_default_162)
        sqrt_default_162 = None
        sub_tensor_162 = torch.ops.aten.sub.Tensor(
            convolution_default_300, mean_dim_162
        )
        mul_tensor_1134 = torch.ops.aten.mul.Tensor(
            sub_tensor_162, reciprocal_default_162
        )
        sub_tensor_162 = None
        squeeze_dim_972 = torch.ops.aten.squeeze.dim(mean_dim_162, 3)
        mean_dim_162 = None
        squeeze_dim_973 = torch.ops.aten.squeeze.dim(squeeze_dim_972, 2)
        squeeze_dim_972 = None
        squeeze_dim_974 = torch.ops.aten.squeeze.dim(squeeze_dim_973, 0)
        squeeze_dim_973 = None
        squeeze_dim_975 = torch.ops.aten.squeeze.dim(reciprocal_default_162, 3)
        reciprocal_default_162 = None
        squeeze_dim_976 = torch.ops.aten.squeeze.dim(squeeze_dim_975, 2)
        squeeze_dim_975 = None
        squeeze_dim_977 = torch.ops.aten.squeeze.dim(squeeze_dim_976, 0)
        squeeze_dim_976 = None
        unsqueeze_default_648 = torch.ops.aten.unsqueeze.default(primals_628, -1)
        unsqueeze_default_649 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_648, -1
        )
        unsqueeze_default_648 = None
        unsqueeze_default_650 = torch.ops.aten.unsqueeze.default(primals_629, -1)
        primals_629 = None
        unsqueeze_default_651 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_650, -1
        )
        unsqueeze_default_650 = None
        mul_tensor_1140 = torch.ops.aten.mul.Tensor(
            mul_tensor_1134, unsqueeze_default_649
        )
        mul_tensor_1134 = unsqueeze_default_649 = None
        add_tensor_718 = torch.ops.aten.add.Tensor(
            mul_tensor_1140, unsqueeze_default_651
        )
        mul_tensor_1140 = unsqueeze_default_651 = None
        relu_default_162 = torch.ops.aten.relu.default(add_tensor_718)
        add_tensor_718 = None
        convolution_default_301 = torch.ops.aten.convolution.default(
            relu_default_162,
            primals_630,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_302 = torch.ops.aten.convolution.default(
            convolution_default_301,
            primals_631,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_326 = torch.ops.aten.var.correction(
            convolution_default_302, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_163 = torch.ops.aten.mean.dim(convolution_default_302, [0, 2, 3], True)
        add_tensor_719 = torch.ops.aten.add.Tensor(var_correction_326, 0.001)
        var_correction_326 = None
        sqrt_default_163 = torch.ops.aten.sqrt.default(add_tensor_719)
        add_tensor_719 = None
        reciprocal_default_163 = torch.ops.aten.reciprocal.default(sqrt_default_163)
        sqrt_default_163 = None
        sub_tensor_163 = torch.ops.aten.sub.Tensor(
            convolution_default_302, mean_dim_163
        )
        mul_tensor_1141 = torch.ops.aten.mul.Tensor(
            sub_tensor_163, reciprocal_default_163
        )
        sub_tensor_163 = None
        squeeze_dim_978 = torch.ops.aten.squeeze.dim(mean_dim_163, 3)
        mean_dim_163 = None
        squeeze_dim_979 = torch.ops.aten.squeeze.dim(squeeze_dim_978, 2)
        squeeze_dim_978 = None
        squeeze_dim_980 = torch.ops.aten.squeeze.dim(squeeze_dim_979, 0)
        squeeze_dim_979 = None
        squeeze_dim_981 = torch.ops.aten.squeeze.dim(reciprocal_default_163, 3)
        reciprocal_default_163 = None
        squeeze_dim_982 = torch.ops.aten.squeeze.dim(squeeze_dim_981, 2)
        squeeze_dim_981 = None
        squeeze_dim_983 = torch.ops.aten.squeeze.dim(squeeze_dim_982, 0)
        squeeze_dim_982 = None
        unsqueeze_default_652 = torch.ops.aten.unsqueeze.default(primals_632, -1)
        unsqueeze_default_653 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_652, -1
        )
        unsqueeze_default_652 = None
        unsqueeze_default_654 = torch.ops.aten.unsqueeze.default(primals_633, -1)
        primals_633 = None
        unsqueeze_default_655 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_654, -1
        )
        unsqueeze_default_654 = None
        mul_tensor_1147 = torch.ops.aten.mul.Tensor(
            mul_tensor_1141, unsqueeze_default_653
        )
        mul_tensor_1141 = unsqueeze_default_653 = None
        add_tensor_722 = torch.ops.aten.add.Tensor(
            mul_tensor_1147, unsqueeze_default_655
        )
        mul_tensor_1147 = unsqueeze_default_655 = None
        convolution_default_303 = torch.ops.aten.convolution.default(
            relu_default_159,
            primals_634,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_159 = None
        convolution_default_304 = torch.ops.aten.convolution.default(
            convolution_default_303,
            primals_635,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_328 = torch.ops.aten.var.correction(
            convolution_default_304, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_164 = torch.ops.aten.mean.dim(convolution_default_304, [0, 2, 3], True)
        add_tensor_723 = torch.ops.aten.add.Tensor(var_correction_328, 0.001)
        var_correction_328 = None
        sqrt_default_164 = torch.ops.aten.sqrt.default(add_tensor_723)
        add_tensor_723 = None
        reciprocal_default_164 = torch.ops.aten.reciprocal.default(sqrt_default_164)
        sqrt_default_164 = None
        sub_tensor_164 = torch.ops.aten.sub.Tensor(
            convolution_default_304, mean_dim_164
        )
        mul_tensor_1148 = torch.ops.aten.mul.Tensor(
            sub_tensor_164, reciprocal_default_164
        )
        sub_tensor_164 = None
        squeeze_dim_984 = torch.ops.aten.squeeze.dim(mean_dim_164, 3)
        mean_dim_164 = None
        squeeze_dim_985 = torch.ops.aten.squeeze.dim(squeeze_dim_984, 2)
        squeeze_dim_984 = None
        squeeze_dim_986 = torch.ops.aten.squeeze.dim(squeeze_dim_985, 0)
        squeeze_dim_985 = None
        squeeze_dim_987 = torch.ops.aten.squeeze.dim(reciprocal_default_164, 3)
        reciprocal_default_164 = None
        squeeze_dim_988 = torch.ops.aten.squeeze.dim(squeeze_dim_987, 2)
        squeeze_dim_987 = None
        squeeze_dim_989 = torch.ops.aten.squeeze.dim(squeeze_dim_988, 0)
        squeeze_dim_988 = None
        unsqueeze_default_656 = torch.ops.aten.unsqueeze.default(primals_636, -1)
        unsqueeze_default_657 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_656, -1
        )
        unsqueeze_default_656 = None
        unsqueeze_default_658 = torch.ops.aten.unsqueeze.default(primals_637, -1)
        primals_637 = None
        unsqueeze_default_659 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_658, -1
        )
        unsqueeze_default_658 = None
        mul_tensor_1154 = torch.ops.aten.mul.Tensor(
            mul_tensor_1148, unsqueeze_default_657
        )
        mul_tensor_1148 = unsqueeze_default_657 = None
        add_tensor_726 = torch.ops.aten.add.Tensor(
            mul_tensor_1154, unsqueeze_default_659
        )
        mul_tensor_1154 = unsqueeze_default_659 = None
        relu_default_164 = torch.ops.aten.relu.default(add_tensor_726)
        add_tensor_726 = None
        convolution_default_305 = torch.ops.aten.convolution.default(
            relu_default_164,
            primals_638,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_306 = torch.ops.aten.convolution.default(
            convolution_default_305,
            primals_639,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_330 = torch.ops.aten.var.correction(
            convolution_default_306, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_165 = torch.ops.aten.mean.dim(convolution_default_306, [0, 2, 3], True)
        add_tensor_727 = torch.ops.aten.add.Tensor(var_correction_330, 0.001)
        var_correction_330 = None
        sqrt_default_165 = torch.ops.aten.sqrt.default(add_tensor_727)
        add_tensor_727 = None
        reciprocal_default_165 = torch.ops.aten.reciprocal.default(sqrt_default_165)
        sqrt_default_165 = None
        sub_tensor_165 = torch.ops.aten.sub.Tensor(
            convolution_default_306, mean_dim_165
        )
        mul_tensor_1155 = torch.ops.aten.mul.Tensor(
            sub_tensor_165, reciprocal_default_165
        )
        sub_tensor_165 = None
        squeeze_dim_990 = torch.ops.aten.squeeze.dim(mean_dim_165, 3)
        mean_dim_165 = None
        squeeze_dim_991 = torch.ops.aten.squeeze.dim(squeeze_dim_990, 2)
        squeeze_dim_990 = None
        squeeze_dim_992 = torch.ops.aten.squeeze.dim(squeeze_dim_991, 0)
        squeeze_dim_991 = None
        squeeze_dim_993 = torch.ops.aten.squeeze.dim(reciprocal_default_165, 3)
        reciprocal_default_165 = None
        squeeze_dim_994 = torch.ops.aten.squeeze.dim(squeeze_dim_993, 2)
        squeeze_dim_993 = None
        squeeze_dim_995 = torch.ops.aten.squeeze.dim(squeeze_dim_994, 0)
        squeeze_dim_994 = None
        unsqueeze_default_660 = torch.ops.aten.unsqueeze.default(primals_640, -1)
        unsqueeze_default_661 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_660, -1
        )
        unsqueeze_default_660 = None
        unsqueeze_default_662 = torch.ops.aten.unsqueeze.default(primals_641, -1)
        primals_641 = None
        unsqueeze_default_663 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_662, -1
        )
        unsqueeze_default_662 = None
        mul_tensor_1161 = torch.ops.aten.mul.Tensor(
            mul_tensor_1155, unsqueeze_default_661
        )
        mul_tensor_1155 = unsqueeze_default_661 = None
        add_tensor_730 = torch.ops.aten.add.Tensor(
            mul_tensor_1161, unsqueeze_default_663
        )
        mul_tensor_1161 = unsqueeze_default_663 = None
        add_tensor_731 = torch.ops.aten.add.Tensor(add_tensor_722, add_tensor_730)
        add_tensor_722 = add_tensor_730 = None
        avg_pool2d_default_42 = torch.ops.aten.avg_pool2d.default(
            add_tensor_697, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_732 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_42, add_tensor_693
        )
        avg_pool2d_default_42 = None
        avg_pool2d_default_43 = torch.ops.aten.avg_pool2d.default(
            add_tensor_693, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_733 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_43, avg_pool2d_default_43
        )
        avg_pool2d_default_43 = None
        convolution_default_307 = torch.ops.aten.convolution.default(
            relu_default_157,
            primals_642,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_157 = None
        convolution_default_308 = torch.ops.aten.convolution.default(
            convolution_default_307,
            primals_643,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_332 = torch.ops.aten.var.correction(
            convolution_default_308, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_166 = torch.ops.aten.mean.dim(convolution_default_308, [0, 2, 3], True)
        add_tensor_734 = torch.ops.aten.add.Tensor(var_correction_332, 0.001)
        var_correction_332 = None
        sqrt_default_166 = torch.ops.aten.sqrt.default(add_tensor_734)
        add_tensor_734 = None
        reciprocal_default_166 = torch.ops.aten.reciprocal.default(sqrt_default_166)
        sqrt_default_166 = None
        sub_tensor_166 = torch.ops.aten.sub.Tensor(
            convolution_default_308, mean_dim_166
        )
        mul_tensor_1162 = torch.ops.aten.mul.Tensor(
            sub_tensor_166, reciprocal_default_166
        )
        sub_tensor_166 = None
        squeeze_dim_996 = torch.ops.aten.squeeze.dim(mean_dim_166, 3)
        mean_dim_166 = None
        squeeze_dim_997 = torch.ops.aten.squeeze.dim(squeeze_dim_996, 2)
        squeeze_dim_996 = None
        squeeze_dim_998 = torch.ops.aten.squeeze.dim(squeeze_dim_997, 0)
        squeeze_dim_997 = None
        squeeze_dim_999 = torch.ops.aten.squeeze.dim(reciprocal_default_166, 3)
        reciprocal_default_166 = None
        squeeze_dim_1000 = torch.ops.aten.squeeze.dim(squeeze_dim_999, 2)
        squeeze_dim_999 = None
        squeeze_dim_1001 = torch.ops.aten.squeeze.dim(squeeze_dim_1000, 0)
        squeeze_dim_1000 = None
        unsqueeze_default_664 = torch.ops.aten.unsqueeze.default(primals_644, -1)
        unsqueeze_default_665 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_664, -1
        )
        unsqueeze_default_664 = None
        unsqueeze_default_666 = torch.ops.aten.unsqueeze.default(primals_645, -1)
        primals_645 = None
        unsqueeze_default_667 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_666, -1
        )
        unsqueeze_default_666 = None
        mul_tensor_1168 = torch.ops.aten.mul.Tensor(
            mul_tensor_1162, unsqueeze_default_665
        )
        mul_tensor_1162 = unsqueeze_default_665 = None
        add_tensor_737 = torch.ops.aten.add.Tensor(
            mul_tensor_1168, unsqueeze_default_667
        )
        mul_tensor_1168 = unsqueeze_default_667 = None
        relu_default_166 = torch.ops.aten.relu.default(add_tensor_737)
        add_tensor_737 = None
        convolution_default_309 = torch.ops.aten.convolution.default(
            relu_default_166,
            primals_646,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_310 = torch.ops.aten.convolution.default(
            convolution_default_309,
            primals_647,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_334 = torch.ops.aten.var.correction(
            convolution_default_310, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_167 = torch.ops.aten.mean.dim(convolution_default_310, [0, 2, 3], True)
        add_tensor_738 = torch.ops.aten.add.Tensor(var_correction_334, 0.001)
        var_correction_334 = None
        sqrt_default_167 = torch.ops.aten.sqrt.default(add_tensor_738)
        add_tensor_738 = None
        reciprocal_default_167 = torch.ops.aten.reciprocal.default(sqrt_default_167)
        sqrt_default_167 = None
        sub_tensor_167 = torch.ops.aten.sub.Tensor(
            convolution_default_310, mean_dim_167
        )
        mul_tensor_1169 = torch.ops.aten.mul.Tensor(
            sub_tensor_167, reciprocal_default_167
        )
        sub_tensor_167 = None
        squeeze_dim_1002 = torch.ops.aten.squeeze.dim(mean_dim_167, 3)
        mean_dim_167 = None
        squeeze_dim_1003 = torch.ops.aten.squeeze.dim(squeeze_dim_1002, 2)
        squeeze_dim_1002 = None
        squeeze_dim_1004 = torch.ops.aten.squeeze.dim(squeeze_dim_1003, 0)
        squeeze_dim_1003 = None
        squeeze_dim_1005 = torch.ops.aten.squeeze.dim(reciprocal_default_167, 3)
        reciprocal_default_167 = None
        squeeze_dim_1006 = torch.ops.aten.squeeze.dim(squeeze_dim_1005, 2)
        squeeze_dim_1005 = None
        squeeze_dim_1007 = torch.ops.aten.squeeze.dim(squeeze_dim_1006, 0)
        squeeze_dim_1006 = None
        unsqueeze_default_668 = torch.ops.aten.unsqueeze.default(primals_648, -1)
        unsqueeze_default_669 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_668, -1
        )
        unsqueeze_default_668 = None
        unsqueeze_default_670 = torch.ops.aten.unsqueeze.default(primals_649, -1)
        primals_649 = None
        unsqueeze_default_671 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_670, -1
        )
        unsqueeze_default_670 = None
        mul_tensor_1175 = torch.ops.aten.mul.Tensor(
            mul_tensor_1169, unsqueeze_default_669
        )
        mul_tensor_1169 = unsqueeze_default_669 = None
        add_tensor_741 = torch.ops.aten.add.Tensor(
            mul_tensor_1175, unsqueeze_default_671
        )
        mul_tensor_1175 = unsqueeze_default_671 = None
        add_tensor_742 = torch.ops.aten.add.Tensor(add_tensor_741, add_tensor_697)
        add_tensor_741 = add_tensor_697 = None
        cat_default_16 = torch.ops.aten.cat.default(
            [
                add_tensor_693,
                add_tensor_714,
                add_tensor_731,
                add_tensor_732,
                add_tensor_733,
                add_tensor_742,
            ],
            1,
        )
        add_tensor_693 = (
            add_tensor_714
        ) = add_tensor_731 = add_tensor_732 = add_tensor_733 = add_tensor_742 = None
        convolution_default_311 = torch.ops.aten.convolution.default(
            relu_default_156,
            primals_650,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_336 = torch.ops.aten.var.correction(
            convolution_default_311, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_168 = torch.ops.aten.mean.dim(convolution_default_311, [0, 2, 3], True)
        add_tensor_743 = torch.ops.aten.add.Tensor(var_correction_336, 0.001)
        var_correction_336 = None
        sqrt_default_168 = torch.ops.aten.sqrt.default(add_tensor_743)
        add_tensor_743 = None
        reciprocal_default_168 = torch.ops.aten.reciprocal.default(sqrt_default_168)
        sqrt_default_168 = None
        sub_tensor_168 = torch.ops.aten.sub.Tensor(
            convolution_default_311, mean_dim_168
        )
        mul_tensor_1176 = torch.ops.aten.mul.Tensor(
            sub_tensor_168, reciprocal_default_168
        )
        sub_tensor_168 = None
        unsqueeze_default_672 = torch.ops.aten.unsqueeze.default(primals_651, -1)
        unsqueeze_default_673 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_672, -1
        )
        unsqueeze_default_672 = None
        unsqueeze_default_674 = torch.ops.aten.unsqueeze.default(primals_652, -1)
        unsqueeze_default_675 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_674, -1
        )
        unsqueeze_default_674 = None
        mul_tensor_1182 = torch.ops.aten.mul.Tensor(
            mul_tensor_1176, unsqueeze_default_673
        )
        mul_tensor_1176 = unsqueeze_default_673 = None
        add_tensor_746 = torch.ops.aten.add.Tensor(
            mul_tensor_1182, unsqueeze_default_675
        )
        mul_tensor_1182 = unsqueeze_default_675 = None
        relu_default_168 = torch.ops.aten.relu.default(cat_default_16)
        cat_default_16 = None
        convolution_default_312 = torch.ops.aten.convolution.default(
            relu_default_168,
            primals_653,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_338 = torch.ops.aten.var.correction(
            convolution_default_312, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_169 = torch.ops.aten.mean.dim(convolution_default_312, [0, 2, 3], True)
        add_tensor_747 = torch.ops.aten.add.Tensor(var_correction_338, 0.001)
        var_correction_338 = None
        sqrt_default_169 = torch.ops.aten.sqrt.default(add_tensor_747)
        add_tensor_747 = None
        reciprocal_default_169 = torch.ops.aten.reciprocal.default(sqrt_default_169)
        sqrt_default_169 = None
        sub_tensor_169 = torch.ops.aten.sub.Tensor(
            convolution_default_312, mean_dim_169
        )
        mul_tensor_1183 = torch.ops.aten.mul.Tensor(
            sub_tensor_169, reciprocal_default_169
        )
        sub_tensor_169 = None
        unsqueeze_default_676 = torch.ops.aten.unsqueeze.default(primals_654, -1)
        unsqueeze_default_677 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_676, -1
        )
        unsqueeze_default_676 = None
        unsqueeze_default_678 = torch.ops.aten.unsqueeze.default(primals_655, -1)
        unsqueeze_default_679 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_678, -1
        )
        unsqueeze_default_678 = None
        mul_tensor_1189 = torch.ops.aten.mul.Tensor(
            mul_tensor_1183, unsqueeze_default_677
        )
        mul_tensor_1183 = unsqueeze_default_677 = None
        add_tensor_750 = torch.ops.aten.add.Tensor(
            mul_tensor_1189, unsqueeze_default_679
        )
        mul_tensor_1189 = unsqueeze_default_679 = None
        relu_default_169 = torch.ops.aten.relu.default(add_tensor_750)
        convolution_default_313 = torch.ops.aten.convolution.default(
            relu_default_169,
            primals_656,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_314 = torch.ops.aten.convolution.default(
            convolution_default_313,
            primals_657,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_340 = torch.ops.aten.var.correction(
            convolution_default_314, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_170 = torch.ops.aten.mean.dim(convolution_default_314, [0, 2, 3], True)
        add_tensor_751 = torch.ops.aten.add.Tensor(var_correction_340, 0.001)
        var_correction_340 = None
        sqrt_default_170 = torch.ops.aten.sqrt.default(add_tensor_751)
        add_tensor_751 = None
        reciprocal_default_170 = torch.ops.aten.reciprocal.default(sqrt_default_170)
        sqrt_default_170 = None
        sub_tensor_170 = torch.ops.aten.sub.Tensor(
            convolution_default_314, mean_dim_170
        )
        mul_tensor_1190 = torch.ops.aten.mul.Tensor(
            sub_tensor_170, reciprocal_default_170
        )
        sub_tensor_170 = None
        squeeze_dim_1020 = torch.ops.aten.squeeze.dim(mean_dim_170, 3)
        mean_dim_170 = None
        squeeze_dim_1021 = torch.ops.aten.squeeze.dim(squeeze_dim_1020, 2)
        squeeze_dim_1020 = None
        squeeze_dim_1022 = torch.ops.aten.squeeze.dim(squeeze_dim_1021, 0)
        squeeze_dim_1021 = None
        squeeze_dim_1023 = torch.ops.aten.squeeze.dim(reciprocal_default_170, 3)
        reciprocal_default_170 = None
        squeeze_dim_1024 = torch.ops.aten.squeeze.dim(squeeze_dim_1023, 2)
        squeeze_dim_1023 = None
        squeeze_dim_1025 = torch.ops.aten.squeeze.dim(squeeze_dim_1024, 0)
        squeeze_dim_1024 = None
        unsqueeze_default_680 = torch.ops.aten.unsqueeze.default(primals_658, -1)
        unsqueeze_default_681 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_680, -1
        )
        unsqueeze_default_680 = None
        unsqueeze_default_682 = torch.ops.aten.unsqueeze.default(primals_659, -1)
        primals_659 = None
        unsqueeze_default_683 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_682, -1
        )
        unsqueeze_default_682 = None
        mul_tensor_1196 = torch.ops.aten.mul.Tensor(
            mul_tensor_1190, unsqueeze_default_681
        )
        mul_tensor_1190 = unsqueeze_default_681 = None
        add_tensor_754 = torch.ops.aten.add.Tensor(
            mul_tensor_1196, unsqueeze_default_683
        )
        mul_tensor_1196 = unsqueeze_default_683 = None
        relu_default_170 = torch.ops.aten.relu.default(add_tensor_754)
        add_tensor_754 = None
        convolution_default_315 = torch.ops.aten.convolution.default(
            relu_default_170,
            primals_660,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_316 = torch.ops.aten.convolution.default(
            convolution_default_315,
            primals_661,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_342 = torch.ops.aten.var.correction(
            convolution_default_316, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_171 = torch.ops.aten.mean.dim(convolution_default_316, [0, 2, 3], True)
        add_tensor_755 = torch.ops.aten.add.Tensor(var_correction_342, 0.001)
        var_correction_342 = None
        sqrt_default_171 = torch.ops.aten.sqrt.default(add_tensor_755)
        add_tensor_755 = None
        reciprocal_default_171 = torch.ops.aten.reciprocal.default(sqrt_default_171)
        sqrt_default_171 = None
        sub_tensor_171 = torch.ops.aten.sub.Tensor(
            convolution_default_316, mean_dim_171
        )
        mul_tensor_1197 = torch.ops.aten.mul.Tensor(
            sub_tensor_171, reciprocal_default_171
        )
        sub_tensor_171 = None
        squeeze_dim_1026 = torch.ops.aten.squeeze.dim(mean_dim_171, 3)
        mean_dim_171 = None
        squeeze_dim_1027 = torch.ops.aten.squeeze.dim(squeeze_dim_1026, 2)
        squeeze_dim_1026 = None
        squeeze_dim_1028 = torch.ops.aten.squeeze.dim(squeeze_dim_1027, 0)
        squeeze_dim_1027 = None
        squeeze_dim_1029 = torch.ops.aten.squeeze.dim(reciprocal_default_171, 3)
        reciprocal_default_171 = None
        squeeze_dim_1030 = torch.ops.aten.squeeze.dim(squeeze_dim_1029, 2)
        squeeze_dim_1029 = None
        squeeze_dim_1031 = torch.ops.aten.squeeze.dim(squeeze_dim_1030, 0)
        squeeze_dim_1030 = None
        unsqueeze_default_684 = torch.ops.aten.unsqueeze.default(primals_662, -1)
        unsqueeze_default_685 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_684, -1
        )
        unsqueeze_default_684 = None
        unsqueeze_default_686 = torch.ops.aten.unsqueeze.default(primals_663, -1)
        primals_663 = None
        unsqueeze_default_687 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_686, -1
        )
        unsqueeze_default_686 = None
        mul_tensor_1203 = torch.ops.aten.mul.Tensor(
            mul_tensor_1197, unsqueeze_default_685
        )
        mul_tensor_1197 = unsqueeze_default_685 = None
        add_tensor_758 = torch.ops.aten.add.Tensor(
            mul_tensor_1203, unsqueeze_default_687
        )
        mul_tensor_1203 = unsqueeze_default_687 = None
        relu_default_171 = torch.ops.aten.relu.default(add_tensor_746)
        convolution_default_317 = torch.ops.aten.convolution.default(
            relu_default_171,
            primals_664,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_318 = torch.ops.aten.convolution.default(
            convolution_default_317,
            primals_665,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_344 = torch.ops.aten.var.correction(
            convolution_default_318, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_172 = torch.ops.aten.mean.dim(convolution_default_318, [0, 2, 3], True)
        add_tensor_759 = torch.ops.aten.add.Tensor(var_correction_344, 0.001)
        var_correction_344 = None
        sqrt_default_172 = torch.ops.aten.sqrt.default(add_tensor_759)
        add_tensor_759 = None
        reciprocal_default_172 = torch.ops.aten.reciprocal.default(sqrt_default_172)
        sqrt_default_172 = None
        sub_tensor_172 = torch.ops.aten.sub.Tensor(
            convolution_default_318, mean_dim_172
        )
        mul_tensor_1204 = torch.ops.aten.mul.Tensor(
            sub_tensor_172, reciprocal_default_172
        )
        sub_tensor_172 = None
        squeeze_dim_1032 = torch.ops.aten.squeeze.dim(mean_dim_172, 3)
        mean_dim_172 = None
        squeeze_dim_1033 = torch.ops.aten.squeeze.dim(squeeze_dim_1032, 2)
        squeeze_dim_1032 = None
        squeeze_dim_1034 = torch.ops.aten.squeeze.dim(squeeze_dim_1033, 0)
        squeeze_dim_1033 = None
        squeeze_dim_1035 = torch.ops.aten.squeeze.dim(reciprocal_default_172, 3)
        reciprocal_default_172 = None
        squeeze_dim_1036 = torch.ops.aten.squeeze.dim(squeeze_dim_1035, 2)
        squeeze_dim_1035 = None
        squeeze_dim_1037 = torch.ops.aten.squeeze.dim(squeeze_dim_1036, 0)
        squeeze_dim_1036 = None
        unsqueeze_default_688 = torch.ops.aten.unsqueeze.default(primals_666, -1)
        unsqueeze_default_689 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_688, -1
        )
        unsqueeze_default_688 = None
        unsqueeze_default_690 = torch.ops.aten.unsqueeze.default(primals_667, -1)
        primals_667 = None
        unsqueeze_default_691 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_690, -1
        )
        unsqueeze_default_690 = None
        mul_tensor_1210 = torch.ops.aten.mul.Tensor(
            mul_tensor_1204, unsqueeze_default_689
        )
        mul_tensor_1204 = unsqueeze_default_689 = None
        add_tensor_762 = torch.ops.aten.add.Tensor(
            mul_tensor_1210, unsqueeze_default_691
        )
        mul_tensor_1210 = unsqueeze_default_691 = None
        relu_default_172 = torch.ops.aten.relu.default(add_tensor_762)
        add_tensor_762 = None
        convolution_default_319 = torch.ops.aten.convolution.default(
            relu_default_172,
            primals_668,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_320 = torch.ops.aten.convolution.default(
            convolution_default_319,
            primals_669,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_346 = torch.ops.aten.var.correction(
            convolution_default_320, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_173 = torch.ops.aten.mean.dim(convolution_default_320, [0, 2, 3], True)
        add_tensor_763 = torch.ops.aten.add.Tensor(var_correction_346, 0.001)
        var_correction_346 = None
        sqrt_default_173 = torch.ops.aten.sqrt.default(add_tensor_763)
        add_tensor_763 = None
        reciprocal_default_173 = torch.ops.aten.reciprocal.default(sqrt_default_173)
        sqrt_default_173 = None
        sub_tensor_173 = torch.ops.aten.sub.Tensor(
            convolution_default_320, mean_dim_173
        )
        mul_tensor_1211 = torch.ops.aten.mul.Tensor(
            sub_tensor_173, reciprocal_default_173
        )
        sub_tensor_173 = None
        squeeze_dim_1038 = torch.ops.aten.squeeze.dim(mean_dim_173, 3)
        mean_dim_173 = None
        squeeze_dim_1039 = torch.ops.aten.squeeze.dim(squeeze_dim_1038, 2)
        squeeze_dim_1038 = None
        squeeze_dim_1040 = torch.ops.aten.squeeze.dim(squeeze_dim_1039, 0)
        squeeze_dim_1039 = None
        squeeze_dim_1041 = torch.ops.aten.squeeze.dim(reciprocal_default_173, 3)
        reciprocal_default_173 = None
        squeeze_dim_1042 = torch.ops.aten.squeeze.dim(squeeze_dim_1041, 2)
        squeeze_dim_1041 = None
        squeeze_dim_1043 = torch.ops.aten.squeeze.dim(squeeze_dim_1042, 0)
        squeeze_dim_1042 = None
        unsqueeze_default_692 = torch.ops.aten.unsqueeze.default(primals_670, -1)
        unsqueeze_default_693 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_692, -1
        )
        unsqueeze_default_692 = None
        unsqueeze_default_694 = torch.ops.aten.unsqueeze.default(primals_671, -1)
        primals_671 = None
        unsqueeze_default_695 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_694, -1
        )
        unsqueeze_default_694 = None
        mul_tensor_1217 = torch.ops.aten.mul.Tensor(
            mul_tensor_1211, unsqueeze_default_693
        )
        mul_tensor_1211 = unsqueeze_default_693 = None
        add_tensor_766 = torch.ops.aten.add.Tensor(
            mul_tensor_1217, unsqueeze_default_695
        )
        mul_tensor_1217 = unsqueeze_default_695 = None
        add_tensor_767 = torch.ops.aten.add.Tensor(add_tensor_758, add_tensor_766)
        add_tensor_758 = add_tensor_766 = None
        convolution_default_321 = torch.ops.aten.convolution.default(
            relu_default_171,
            primals_672,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_322 = torch.ops.aten.convolution.default(
            convolution_default_321,
            primals_673,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_348 = torch.ops.aten.var.correction(
            convolution_default_322, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_174 = torch.ops.aten.mean.dim(convolution_default_322, [0, 2, 3], True)
        add_tensor_768 = torch.ops.aten.add.Tensor(var_correction_348, 0.001)
        var_correction_348 = None
        sqrt_default_174 = torch.ops.aten.sqrt.default(add_tensor_768)
        add_tensor_768 = None
        reciprocal_default_174 = torch.ops.aten.reciprocal.default(sqrt_default_174)
        sqrt_default_174 = None
        sub_tensor_174 = torch.ops.aten.sub.Tensor(
            convolution_default_322, mean_dim_174
        )
        mul_tensor_1218 = torch.ops.aten.mul.Tensor(
            sub_tensor_174, reciprocal_default_174
        )
        sub_tensor_174 = None
        squeeze_dim_1044 = torch.ops.aten.squeeze.dim(mean_dim_174, 3)
        mean_dim_174 = None
        squeeze_dim_1045 = torch.ops.aten.squeeze.dim(squeeze_dim_1044, 2)
        squeeze_dim_1044 = None
        squeeze_dim_1046 = torch.ops.aten.squeeze.dim(squeeze_dim_1045, 0)
        squeeze_dim_1045 = None
        squeeze_dim_1047 = torch.ops.aten.squeeze.dim(reciprocal_default_174, 3)
        reciprocal_default_174 = None
        squeeze_dim_1048 = torch.ops.aten.squeeze.dim(squeeze_dim_1047, 2)
        squeeze_dim_1047 = None
        squeeze_dim_1049 = torch.ops.aten.squeeze.dim(squeeze_dim_1048, 0)
        squeeze_dim_1048 = None
        unsqueeze_default_696 = torch.ops.aten.unsqueeze.default(primals_674, -1)
        unsqueeze_default_697 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_696, -1
        )
        unsqueeze_default_696 = None
        unsqueeze_default_698 = torch.ops.aten.unsqueeze.default(primals_675, -1)
        primals_675 = None
        unsqueeze_default_699 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_698, -1
        )
        unsqueeze_default_698 = None
        mul_tensor_1224 = torch.ops.aten.mul.Tensor(
            mul_tensor_1218, unsqueeze_default_697
        )
        mul_tensor_1218 = unsqueeze_default_697 = None
        add_tensor_771 = torch.ops.aten.add.Tensor(
            mul_tensor_1224, unsqueeze_default_699
        )
        mul_tensor_1224 = unsqueeze_default_699 = None
        relu_default_174 = torch.ops.aten.relu.default(add_tensor_771)
        add_tensor_771 = None
        convolution_default_323 = torch.ops.aten.convolution.default(
            relu_default_174,
            primals_676,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_324 = torch.ops.aten.convolution.default(
            convolution_default_323,
            primals_677,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_350 = torch.ops.aten.var.correction(
            convolution_default_324, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_175 = torch.ops.aten.mean.dim(convolution_default_324, [0, 2, 3], True)
        add_tensor_772 = torch.ops.aten.add.Tensor(var_correction_350, 0.001)
        var_correction_350 = None
        sqrt_default_175 = torch.ops.aten.sqrt.default(add_tensor_772)
        add_tensor_772 = None
        reciprocal_default_175 = torch.ops.aten.reciprocal.default(sqrt_default_175)
        sqrt_default_175 = None
        sub_tensor_175 = torch.ops.aten.sub.Tensor(
            convolution_default_324, mean_dim_175
        )
        mul_tensor_1225 = torch.ops.aten.mul.Tensor(
            sub_tensor_175, reciprocal_default_175
        )
        sub_tensor_175 = None
        squeeze_dim_1050 = torch.ops.aten.squeeze.dim(mean_dim_175, 3)
        mean_dim_175 = None
        squeeze_dim_1051 = torch.ops.aten.squeeze.dim(squeeze_dim_1050, 2)
        squeeze_dim_1050 = None
        squeeze_dim_1052 = torch.ops.aten.squeeze.dim(squeeze_dim_1051, 0)
        squeeze_dim_1051 = None
        squeeze_dim_1053 = torch.ops.aten.squeeze.dim(reciprocal_default_175, 3)
        reciprocal_default_175 = None
        squeeze_dim_1054 = torch.ops.aten.squeeze.dim(squeeze_dim_1053, 2)
        squeeze_dim_1053 = None
        squeeze_dim_1055 = torch.ops.aten.squeeze.dim(squeeze_dim_1054, 0)
        squeeze_dim_1054 = None
        unsqueeze_default_700 = torch.ops.aten.unsqueeze.default(primals_678, -1)
        unsqueeze_default_701 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_700, -1
        )
        unsqueeze_default_700 = None
        unsqueeze_default_702 = torch.ops.aten.unsqueeze.default(primals_679, -1)
        primals_679 = None
        unsqueeze_default_703 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_702, -1
        )
        unsqueeze_default_702 = None
        mul_tensor_1231 = torch.ops.aten.mul.Tensor(
            mul_tensor_1225, unsqueeze_default_701
        )
        mul_tensor_1225 = unsqueeze_default_701 = None
        add_tensor_775 = torch.ops.aten.add.Tensor(
            mul_tensor_1231, unsqueeze_default_703
        )
        mul_tensor_1231 = unsqueeze_default_703 = None
        convolution_default_325 = torch.ops.aten.convolution.default(
            relu_default_171,
            primals_680,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_171 = None
        convolution_default_326 = torch.ops.aten.convolution.default(
            convolution_default_325,
            primals_681,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_352 = torch.ops.aten.var.correction(
            convolution_default_326, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_176 = torch.ops.aten.mean.dim(convolution_default_326, [0, 2, 3], True)
        add_tensor_776 = torch.ops.aten.add.Tensor(var_correction_352, 0.001)
        var_correction_352 = None
        sqrt_default_176 = torch.ops.aten.sqrt.default(add_tensor_776)
        add_tensor_776 = None
        reciprocal_default_176 = torch.ops.aten.reciprocal.default(sqrt_default_176)
        sqrt_default_176 = None
        sub_tensor_176 = torch.ops.aten.sub.Tensor(
            convolution_default_326, mean_dim_176
        )
        mul_tensor_1232 = torch.ops.aten.mul.Tensor(
            sub_tensor_176, reciprocal_default_176
        )
        sub_tensor_176 = None
        squeeze_dim_1056 = torch.ops.aten.squeeze.dim(mean_dim_176, 3)
        mean_dim_176 = None
        squeeze_dim_1057 = torch.ops.aten.squeeze.dim(squeeze_dim_1056, 2)
        squeeze_dim_1056 = None
        squeeze_dim_1058 = torch.ops.aten.squeeze.dim(squeeze_dim_1057, 0)
        squeeze_dim_1057 = None
        squeeze_dim_1059 = torch.ops.aten.squeeze.dim(reciprocal_default_176, 3)
        reciprocal_default_176 = None
        squeeze_dim_1060 = torch.ops.aten.squeeze.dim(squeeze_dim_1059, 2)
        squeeze_dim_1059 = None
        squeeze_dim_1061 = torch.ops.aten.squeeze.dim(squeeze_dim_1060, 0)
        squeeze_dim_1060 = None
        unsqueeze_default_704 = torch.ops.aten.unsqueeze.default(primals_682, -1)
        unsqueeze_default_705 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_704, -1
        )
        unsqueeze_default_704 = None
        unsqueeze_default_706 = torch.ops.aten.unsqueeze.default(primals_683, -1)
        primals_683 = None
        unsqueeze_default_707 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_706, -1
        )
        unsqueeze_default_706 = None
        mul_tensor_1238 = torch.ops.aten.mul.Tensor(
            mul_tensor_1232, unsqueeze_default_705
        )
        mul_tensor_1232 = unsqueeze_default_705 = None
        add_tensor_779 = torch.ops.aten.add.Tensor(
            mul_tensor_1238, unsqueeze_default_707
        )
        mul_tensor_1238 = unsqueeze_default_707 = None
        relu_default_176 = torch.ops.aten.relu.default(add_tensor_779)
        add_tensor_779 = None
        convolution_default_327 = torch.ops.aten.convolution.default(
            relu_default_176,
            primals_684,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_328 = torch.ops.aten.convolution.default(
            convolution_default_327,
            primals_685,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_354 = torch.ops.aten.var.correction(
            convolution_default_328, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_177 = torch.ops.aten.mean.dim(convolution_default_328, [0, 2, 3], True)
        add_tensor_780 = torch.ops.aten.add.Tensor(var_correction_354, 0.001)
        var_correction_354 = None
        sqrt_default_177 = torch.ops.aten.sqrt.default(add_tensor_780)
        add_tensor_780 = None
        reciprocal_default_177 = torch.ops.aten.reciprocal.default(sqrt_default_177)
        sqrt_default_177 = None
        sub_tensor_177 = torch.ops.aten.sub.Tensor(
            convolution_default_328, mean_dim_177
        )
        mul_tensor_1239 = torch.ops.aten.mul.Tensor(
            sub_tensor_177, reciprocal_default_177
        )
        sub_tensor_177 = None
        squeeze_dim_1062 = torch.ops.aten.squeeze.dim(mean_dim_177, 3)
        mean_dim_177 = None
        squeeze_dim_1063 = torch.ops.aten.squeeze.dim(squeeze_dim_1062, 2)
        squeeze_dim_1062 = None
        squeeze_dim_1064 = torch.ops.aten.squeeze.dim(squeeze_dim_1063, 0)
        squeeze_dim_1063 = None
        squeeze_dim_1065 = torch.ops.aten.squeeze.dim(reciprocal_default_177, 3)
        reciprocal_default_177 = None
        squeeze_dim_1066 = torch.ops.aten.squeeze.dim(squeeze_dim_1065, 2)
        squeeze_dim_1065 = None
        squeeze_dim_1067 = torch.ops.aten.squeeze.dim(squeeze_dim_1066, 0)
        squeeze_dim_1066 = None
        unsqueeze_default_708 = torch.ops.aten.unsqueeze.default(primals_686, -1)
        unsqueeze_default_709 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_708, -1
        )
        unsqueeze_default_708 = None
        unsqueeze_default_710 = torch.ops.aten.unsqueeze.default(primals_687, -1)
        primals_687 = None
        unsqueeze_default_711 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_710, -1
        )
        unsqueeze_default_710 = None
        mul_tensor_1245 = torch.ops.aten.mul.Tensor(
            mul_tensor_1239, unsqueeze_default_709
        )
        mul_tensor_1239 = unsqueeze_default_709 = None
        add_tensor_783 = torch.ops.aten.add.Tensor(
            mul_tensor_1245, unsqueeze_default_711
        )
        mul_tensor_1245 = unsqueeze_default_711 = None
        add_tensor_784 = torch.ops.aten.add.Tensor(add_tensor_775, add_tensor_783)
        add_tensor_775 = add_tensor_783 = None
        avg_pool2d_default_45 = torch.ops.aten.avg_pool2d.default(
            add_tensor_750, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_785 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_45, add_tensor_746
        )
        avg_pool2d_default_45 = None
        avg_pool2d_default_46 = torch.ops.aten.avg_pool2d.default(
            add_tensor_746, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_786 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_46, avg_pool2d_default_46
        )
        avg_pool2d_default_46 = None
        convolution_default_329 = torch.ops.aten.convolution.default(
            relu_default_169,
            primals_688,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        relu_default_169 = None
        convolution_default_330 = torch.ops.aten.convolution.default(
            convolution_default_329,
            primals_689,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_356 = torch.ops.aten.var.correction(
            convolution_default_330, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_178 = torch.ops.aten.mean.dim(convolution_default_330, [0, 2, 3], True)
        add_tensor_787 = torch.ops.aten.add.Tensor(var_correction_356, 0.001)
        var_correction_356 = None
        sqrt_default_178 = torch.ops.aten.sqrt.default(add_tensor_787)
        add_tensor_787 = None
        reciprocal_default_178 = torch.ops.aten.reciprocal.default(sqrt_default_178)
        sqrt_default_178 = None
        sub_tensor_178 = torch.ops.aten.sub.Tensor(
            convolution_default_330, mean_dim_178
        )
        mul_tensor_1246 = torch.ops.aten.mul.Tensor(
            sub_tensor_178, reciprocal_default_178
        )
        sub_tensor_178 = None
        squeeze_dim_1068 = torch.ops.aten.squeeze.dim(mean_dim_178, 3)
        mean_dim_178 = None
        squeeze_dim_1069 = torch.ops.aten.squeeze.dim(squeeze_dim_1068, 2)
        squeeze_dim_1068 = None
        squeeze_dim_1070 = torch.ops.aten.squeeze.dim(squeeze_dim_1069, 0)
        squeeze_dim_1069 = None
        squeeze_dim_1071 = torch.ops.aten.squeeze.dim(reciprocal_default_178, 3)
        reciprocal_default_178 = None
        squeeze_dim_1072 = torch.ops.aten.squeeze.dim(squeeze_dim_1071, 2)
        squeeze_dim_1071 = None
        squeeze_dim_1073 = torch.ops.aten.squeeze.dim(squeeze_dim_1072, 0)
        squeeze_dim_1072 = None
        unsqueeze_default_712 = torch.ops.aten.unsqueeze.default(primals_690, -1)
        unsqueeze_default_713 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_712, -1
        )
        unsqueeze_default_712 = None
        unsqueeze_default_714 = torch.ops.aten.unsqueeze.default(primals_691, -1)
        primals_691 = None
        unsqueeze_default_715 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_714, -1
        )
        unsqueeze_default_714 = None
        mul_tensor_1252 = torch.ops.aten.mul.Tensor(
            mul_tensor_1246, unsqueeze_default_713
        )
        mul_tensor_1246 = unsqueeze_default_713 = None
        add_tensor_790 = torch.ops.aten.add.Tensor(
            mul_tensor_1252, unsqueeze_default_715
        )
        mul_tensor_1252 = unsqueeze_default_715 = None
        relu_default_178 = torch.ops.aten.relu.default(add_tensor_790)
        add_tensor_790 = None
        convolution_default_331 = torch.ops.aten.convolution.default(
            relu_default_178,
            primals_692,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            336,
        )
        convolution_default_332 = torch.ops.aten.convolution.default(
            convolution_default_331,
            primals_693,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_358 = torch.ops.aten.var.correction(
            convolution_default_332, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_179 = torch.ops.aten.mean.dim(convolution_default_332, [0, 2, 3], True)
        add_tensor_791 = torch.ops.aten.add.Tensor(var_correction_358, 0.001)
        var_correction_358 = None
        sqrt_default_179 = torch.ops.aten.sqrt.default(add_tensor_791)
        add_tensor_791 = None
        reciprocal_default_179 = torch.ops.aten.reciprocal.default(sqrt_default_179)
        sqrt_default_179 = None
        sub_tensor_179 = torch.ops.aten.sub.Tensor(
            convolution_default_332, mean_dim_179
        )
        mul_tensor_1253 = torch.ops.aten.mul.Tensor(
            sub_tensor_179, reciprocal_default_179
        )
        sub_tensor_179 = None
        squeeze_dim_1074 = torch.ops.aten.squeeze.dim(mean_dim_179, 3)
        mean_dim_179 = None
        squeeze_dim_1075 = torch.ops.aten.squeeze.dim(squeeze_dim_1074, 2)
        squeeze_dim_1074 = None
        squeeze_dim_1076 = torch.ops.aten.squeeze.dim(squeeze_dim_1075, 0)
        squeeze_dim_1075 = None
        squeeze_dim_1077 = torch.ops.aten.squeeze.dim(reciprocal_default_179, 3)
        reciprocal_default_179 = None
        squeeze_dim_1078 = torch.ops.aten.squeeze.dim(squeeze_dim_1077, 2)
        squeeze_dim_1077 = None
        squeeze_dim_1079 = torch.ops.aten.squeeze.dim(squeeze_dim_1078, 0)
        squeeze_dim_1078 = None
        unsqueeze_default_716 = torch.ops.aten.unsqueeze.default(primals_694, -1)
        unsqueeze_default_717 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_716, -1
        )
        unsqueeze_default_716 = None
        unsqueeze_default_718 = torch.ops.aten.unsqueeze.default(primals_695, -1)
        primals_695 = None
        unsqueeze_default_719 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_718, -1
        )
        unsqueeze_default_718 = None
        mul_tensor_1259 = torch.ops.aten.mul.Tensor(
            mul_tensor_1253, unsqueeze_default_717
        )
        mul_tensor_1253 = unsqueeze_default_717 = None
        add_tensor_794 = torch.ops.aten.add.Tensor(
            mul_tensor_1259, unsqueeze_default_719
        )
        mul_tensor_1259 = unsqueeze_default_719 = None
        add_tensor_795 = torch.ops.aten.add.Tensor(add_tensor_794, add_tensor_750)
        add_tensor_794 = add_tensor_750 = None
        cat_default_17 = torch.ops.aten.cat.default(
            [
                add_tensor_746,
                add_tensor_767,
                add_tensor_784,
                add_tensor_785,
                add_tensor_786,
                add_tensor_795,
            ],
            1,
        )
        add_tensor_746 = (
            add_tensor_767
        ) = add_tensor_784 = add_tensor_785 = add_tensor_786 = add_tensor_795 = None
        convolution_default_333 = torch.ops.aten.convolution.default(
            relu_default_168,
            primals_696,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_360 = torch.ops.aten.var.correction(
            convolution_default_333, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_180 = torch.ops.aten.mean.dim(convolution_default_333, [0, 2, 3], True)
        add_tensor_796 = torch.ops.aten.add.Tensor(var_correction_360, 0.001)
        var_correction_360 = None
        sqrt_default_180 = torch.ops.aten.sqrt.default(add_tensor_796)
        add_tensor_796 = None
        reciprocal_default_180 = torch.ops.aten.reciprocal.default(sqrt_default_180)
        sqrt_default_180 = None
        sub_tensor_180 = torch.ops.aten.sub.Tensor(
            convolution_default_333, mean_dim_180
        )
        mul_tensor_1260 = torch.ops.aten.mul.Tensor(
            sub_tensor_180, reciprocal_default_180
        )
        sub_tensor_180 = None
        squeeze_dim_1080 = torch.ops.aten.squeeze.dim(mean_dim_180, 3)
        mean_dim_180 = None
        squeeze_dim_1081 = torch.ops.aten.squeeze.dim(squeeze_dim_1080, 2)
        squeeze_dim_1080 = None
        squeeze_dim_1082 = torch.ops.aten.squeeze.dim(squeeze_dim_1081, 0)
        squeeze_dim_1081 = None
        squeeze_dim_1083 = torch.ops.aten.squeeze.dim(reciprocal_default_180, 3)
        reciprocal_default_180 = None
        squeeze_dim_1084 = torch.ops.aten.squeeze.dim(squeeze_dim_1083, 2)
        squeeze_dim_1083 = None
        squeeze_dim_1085 = torch.ops.aten.squeeze.dim(squeeze_dim_1084, 0)
        squeeze_dim_1084 = None
        unsqueeze_default_720 = torch.ops.aten.unsqueeze.default(primals_697, -1)
        unsqueeze_default_721 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_720, -1
        )
        unsqueeze_default_720 = None
        unsqueeze_default_722 = torch.ops.aten.unsqueeze.default(primals_698, -1)
        primals_698 = None
        unsqueeze_default_723 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_722, -1
        )
        unsqueeze_default_722 = None
        mul_tensor_1266 = torch.ops.aten.mul.Tensor(
            mul_tensor_1260, unsqueeze_default_721
        )
        mul_tensor_1260 = unsqueeze_default_721 = None
        add_tensor_799 = torch.ops.aten.add.Tensor(
            mul_tensor_1266, unsqueeze_default_723
        )
        mul_tensor_1266 = unsqueeze_default_723 = None
        relu_default_180 = torch.ops.aten.relu.default(cat_default_17)
        cat_default_17 = None
        convolution_default_334 = torch.ops.aten.convolution.default(
            relu_default_180,
            primals_699,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_362 = torch.ops.aten.var.correction(
            convolution_default_334, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_181 = torch.ops.aten.mean.dim(convolution_default_334, [0, 2, 3], True)
        add_tensor_800 = torch.ops.aten.add.Tensor(var_correction_362, 0.001)
        var_correction_362 = None
        sqrt_default_181 = torch.ops.aten.sqrt.default(add_tensor_800)
        add_tensor_800 = None
        reciprocal_default_181 = torch.ops.aten.reciprocal.default(sqrt_default_181)
        sqrt_default_181 = None
        sub_tensor_181 = torch.ops.aten.sub.Tensor(
            convolution_default_334, mean_dim_181
        )
        mul_tensor_1267 = torch.ops.aten.mul.Tensor(
            sub_tensor_181, reciprocal_default_181
        )
        sub_tensor_181 = None
        squeeze_dim_1086 = torch.ops.aten.squeeze.dim(mean_dim_181, 3)
        mean_dim_181 = None
        squeeze_dim_1087 = torch.ops.aten.squeeze.dim(squeeze_dim_1086, 2)
        squeeze_dim_1086 = None
        squeeze_dim_1088 = torch.ops.aten.squeeze.dim(squeeze_dim_1087, 0)
        squeeze_dim_1087 = None
        squeeze_dim_1089 = torch.ops.aten.squeeze.dim(reciprocal_default_181, 3)
        reciprocal_default_181 = None
        squeeze_dim_1090 = torch.ops.aten.squeeze.dim(squeeze_dim_1089, 2)
        squeeze_dim_1089 = None
        squeeze_dim_1091 = torch.ops.aten.squeeze.dim(squeeze_dim_1090, 0)
        squeeze_dim_1090 = None
        unsqueeze_default_724 = torch.ops.aten.unsqueeze.default(primals_700, -1)
        unsqueeze_default_725 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_724, -1
        )
        unsqueeze_default_724 = None
        unsqueeze_default_726 = torch.ops.aten.unsqueeze.default(primals_701, -1)
        primals_701 = None
        unsqueeze_default_727 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_726, -1
        )
        unsqueeze_default_726 = None
        mul_tensor_1273 = torch.ops.aten.mul.Tensor(
            mul_tensor_1267, unsqueeze_default_725
        )
        mul_tensor_1267 = unsqueeze_default_725 = None
        add_tensor_803 = torch.ops.aten.add.Tensor(
            mul_tensor_1273, unsqueeze_default_727
        )
        mul_tensor_1273 = unsqueeze_default_727 = None
        relu_default_181 = torch.ops.aten.relu.default(add_tensor_803)
        constant_pad_nd_default_24 = torch.ops.aten.constant_pad_nd.default(
            relu_default_181, [2, 2, 2, 2], 0.0
        )
        convolution_default_335 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_24,
            primals_13,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_336 = torch.ops.aten.convolution.default(
            convolution_default_335,
            primals_702,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_364 = torch.ops.aten.var.correction(
            convolution_default_336, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_182 = torch.ops.aten.mean.dim(convolution_default_336, [0, 2, 3], True)
        add_tensor_804 = torch.ops.aten.add.Tensor(var_correction_364, 0.001)
        var_correction_364 = None
        sqrt_default_182 = torch.ops.aten.sqrt.default(add_tensor_804)
        add_tensor_804 = None
        reciprocal_default_182 = torch.ops.aten.reciprocal.default(sqrt_default_182)
        sqrt_default_182 = None
        sub_tensor_182 = torch.ops.aten.sub.Tensor(
            convolution_default_336, mean_dim_182
        )
        mul_tensor_1274 = torch.ops.aten.mul.Tensor(
            sub_tensor_182, reciprocal_default_182
        )
        sub_tensor_182 = None
        squeeze_dim_1092 = torch.ops.aten.squeeze.dim(mean_dim_182, 3)
        mean_dim_182 = None
        squeeze_dim_1093 = torch.ops.aten.squeeze.dim(squeeze_dim_1092, 2)
        squeeze_dim_1092 = None
        squeeze_dim_1094 = torch.ops.aten.squeeze.dim(squeeze_dim_1093, 0)
        squeeze_dim_1093 = None
        squeeze_dim_1095 = torch.ops.aten.squeeze.dim(reciprocal_default_182, 3)
        reciprocal_default_182 = None
        squeeze_dim_1096 = torch.ops.aten.squeeze.dim(squeeze_dim_1095, 2)
        squeeze_dim_1095 = None
        squeeze_dim_1097 = torch.ops.aten.squeeze.dim(squeeze_dim_1096, 0)
        squeeze_dim_1096 = None
        unsqueeze_default_728 = torch.ops.aten.unsqueeze.default(primals_703, -1)
        unsqueeze_default_729 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_728, -1
        )
        unsqueeze_default_728 = None
        unsqueeze_default_730 = torch.ops.aten.unsqueeze.default(primals_704, -1)
        primals_704 = None
        unsqueeze_default_731 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_730, -1
        )
        unsqueeze_default_730 = None
        mul_tensor_1280 = torch.ops.aten.mul.Tensor(
            mul_tensor_1274, unsqueeze_default_729
        )
        mul_tensor_1274 = unsqueeze_default_729 = None
        add_tensor_807 = torch.ops.aten.add.Tensor(
            mul_tensor_1280, unsqueeze_default_731
        )
        mul_tensor_1280 = unsqueeze_default_731 = None
        relu_default_182 = torch.ops.aten.relu.default(add_tensor_807)
        add_tensor_807 = None
        convolution_default_337 = torch.ops.aten.convolution.default(
            relu_default_182,
            primals_705,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_338 = torch.ops.aten.convolution.default(
            convolution_default_337,
            primals_706,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_366 = torch.ops.aten.var.correction(
            convolution_default_338, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_183 = torch.ops.aten.mean.dim(convolution_default_338, [0, 2, 3], True)
        add_tensor_808 = torch.ops.aten.add.Tensor(var_correction_366, 0.001)
        var_correction_366 = None
        sqrt_default_183 = torch.ops.aten.sqrt.default(add_tensor_808)
        add_tensor_808 = None
        reciprocal_default_183 = torch.ops.aten.reciprocal.default(sqrt_default_183)
        sqrt_default_183 = None
        sub_tensor_183 = torch.ops.aten.sub.Tensor(
            convolution_default_338, mean_dim_183
        )
        mul_tensor_1281 = torch.ops.aten.mul.Tensor(
            sub_tensor_183, reciprocal_default_183
        )
        sub_tensor_183 = None
        squeeze_dim_1098 = torch.ops.aten.squeeze.dim(mean_dim_183, 3)
        mean_dim_183 = None
        squeeze_dim_1099 = torch.ops.aten.squeeze.dim(squeeze_dim_1098, 2)
        squeeze_dim_1098 = None
        squeeze_dim_1100 = torch.ops.aten.squeeze.dim(squeeze_dim_1099, 0)
        squeeze_dim_1099 = None
        squeeze_dim_1101 = torch.ops.aten.squeeze.dim(reciprocal_default_183, 3)
        reciprocal_default_183 = None
        squeeze_dim_1102 = torch.ops.aten.squeeze.dim(squeeze_dim_1101, 2)
        squeeze_dim_1101 = None
        squeeze_dim_1103 = torch.ops.aten.squeeze.dim(squeeze_dim_1102, 0)
        squeeze_dim_1102 = None
        unsqueeze_default_732 = torch.ops.aten.unsqueeze.default(primals_707, -1)
        unsqueeze_default_733 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_732, -1
        )
        unsqueeze_default_732 = None
        unsqueeze_default_734 = torch.ops.aten.unsqueeze.default(primals_708, -1)
        primals_708 = None
        unsqueeze_default_735 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_734, -1
        )
        unsqueeze_default_734 = None
        mul_tensor_1287 = torch.ops.aten.mul.Tensor(
            mul_tensor_1281, unsqueeze_default_733
        )
        mul_tensor_1281 = unsqueeze_default_733 = None
        add_tensor_811 = torch.ops.aten.add.Tensor(
            mul_tensor_1287, unsqueeze_default_735
        )
        mul_tensor_1287 = unsqueeze_default_735 = None
        relu_default_183 = torch.ops.aten.relu.default(add_tensor_799)
        add_tensor_799 = None
        constant_pad_nd_default_25 = torch.ops.aten.constant_pad_nd.default(
            relu_default_183, [3, 3, 3, 3], 0.0
        )
        convolution_default_339 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_25,
            primals_14,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_340 = torch.ops.aten.convolution.default(
            convolution_default_339,
            primals_709,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_368 = torch.ops.aten.var.correction(
            convolution_default_340, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_184 = torch.ops.aten.mean.dim(convolution_default_340, [0, 2, 3], True)
        add_tensor_812 = torch.ops.aten.add.Tensor(var_correction_368, 0.001)
        var_correction_368 = None
        sqrt_default_184 = torch.ops.aten.sqrt.default(add_tensor_812)
        add_tensor_812 = None
        reciprocal_default_184 = torch.ops.aten.reciprocal.default(sqrt_default_184)
        sqrt_default_184 = None
        sub_tensor_184 = torch.ops.aten.sub.Tensor(
            convolution_default_340, mean_dim_184
        )
        mul_tensor_1288 = torch.ops.aten.mul.Tensor(
            sub_tensor_184, reciprocal_default_184
        )
        sub_tensor_184 = None
        squeeze_dim_1104 = torch.ops.aten.squeeze.dim(mean_dim_184, 3)
        mean_dim_184 = None
        squeeze_dim_1105 = torch.ops.aten.squeeze.dim(squeeze_dim_1104, 2)
        squeeze_dim_1104 = None
        squeeze_dim_1106 = torch.ops.aten.squeeze.dim(squeeze_dim_1105, 0)
        squeeze_dim_1105 = None
        squeeze_dim_1107 = torch.ops.aten.squeeze.dim(reciprocal_default_184, 3)
        reciprocal_default_184 = None
        squeeze_dim_1108 = torch.ops.aten.squeeze.dim(squeeze_dim_1107, 2)
        squeeze_dim_1107 = None
        squeeze_dim_1109 = torch.ops.aten.squeeze.dim(squeeze_dim_1108, 0)
        squeeze_dim_1108 = None
        unsqueeze_default_736 = torch.ops.aten.unsqueeze.default(primals_710, -1)
        unsqueeze_default_737 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_736, -1
        )
        unsqueeze_default_736 = None
        unsqueeze_default_738 = torch.ops.aten.unsqueeze.default(primals_711, -1)
        primals_711 = None
        unsqueeze_default_739 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_738, -1
        )
        unsqueeze_default_738 = None
        mul_tensor_1294 = torch.ops.aten.mul.Tensor(
            mul_tensor_1288, unsqueeze_default_737
        )
        mul_tensor_1288 = unsqueeze_default_737 = None
        add_tensor_815 = torch.ops.aten.add.Tensor(
            mul_tensor_1294, unsqueeze_default_739
        )
        mul_tensor_1294 = unsqueeze_default_739 = None
        relu_default_184 = torch.ops.aten.relu.default(add_tensor_815)
        add_tensor_815 = None
        convolution_default_341 = torch.ops.aten.convolution.default(
            relu_default_184,
            primals_712,
            None,
            [1, 1],
            [3, 3],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_342 = torch.ops.aten.convolution.default(
            convolution_default_341,
            primals_713,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_370 = torch.ops.aten.var.correction(
            convolution_default_342, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_185 = torch.ops.aten.mean.dim(convolution_default_342, [0, 2, 3], True)
        add_tensor_816 = torch.ops.aten.add.Tensor(var_correction_370, 0.001)
        var_correction_370 = None
        sqrt_default_185 = torch.ops.aten.sqrt.default(add_tensor_816)
        add_tensor_816 = None
        reciprocal_default_185 = torch.ops.aten.reciprocal.default(sqrt_default_185)
        sqrt_default_185 = None
        sub_tensor_185 = torch.ops.aten.sub.Tensor(
            convolution_default_342, mean_dim_185
        )
        mul_tensor_1295 = torch.ops.aten.mul.Tensor(
            sub_tensor_185, reciprocal_default_185
        )
        sub_tensor_185 = None
        squeeze_dim_1110 = torch.ops.aten.squeeze.dim(mean_dim_185, 3)
        mean_dim_185 = None
        squeeze_dim_1111 = torch.ops.aten.squeeze.dim(squeeze_dim_1110, 2)
        squeeze_dim_1110 = None
        squeeze_dim_1112 = torch.ops.aten.squeeze.dim(squeeze_dim_1111, 0)
        squeeze_dim_1111 = None
        squeeze_dim_1113 = torch.ops.aten.squeeze.dim(reciprocal_default_185, 3)
        reciprocal_default_185 = None
        squeeze_dim_1114 = torch.ops.aten.squeeze.dim(squeeze_dim_1113, 2)
        squeeze_dim_1113 = None
        squeeze_dim_1115 = torch.ops.aten.squeeze.dim(squeeze_dim_1114, 0)
        squeeze_dim_1114 = None
        unsqueeze_default_740 = torch.ops.aten.unsqueeze.default(primals_714, -1)
        unsqueeze_default_741 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_740, -1
        )
        unsqueeze_default_740 = None
        unsqueeze_default_742 = torch.ops.aten.unsqueeze.default(primals_715, -1)
        primals_715 = None
        unsqueeze_default_743 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_742, -1
        )
        unsqueeze_default_742 = None
        mul_tensor_1301 = torch.ops.aten.mul.Tensor(
            mul_tensor_1295, unsqueeze_default_741
        )
        mul_tensor_1295 = unsqueeze_default_741 = None
        add_tensor_819 = torch.ops.aten.add.Tensor(
            mul_tensor_1301, unsqueeze_default_743
        )
        mul_tensor_1301 = unsqueeze_default_743 = None
        add_tensor_820 = torch.ops.aten.add.Tensor(add_tensor_811, add_tensor_819)
        add_tensor_811 = add_tensor_819 = None
        constant_pad_nd_default_26 = torch.ops.aten.constant_pad_nd.default(
            add_tensor_803, [1, 1, 1, 1], -inf
        )
        max_pool2d_with_indices_default_6 = (
            torch.ops.aten.max_pool2d_with_indices.default(
                constant_pad_nd_default_26, [3, 3], [2, 2]
            )
        )
        getitem_12 = max_pool2d_with_indices_default_6[0]
        getitem_13 = max_pool2d_with_indices_default_6[1]
        max_pool2d_with_indices_default_6 = None
        convolution_default_343 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_25,
            primals_15,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_344 = torch.ops.aten.convolution.default(
            convolution_default_343,
            primals_716,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_372 = torch.ops.aten.var.correction(
            convolution_default_344, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_186 = torch.ops.aten.mean.dim(convolution_default_344, [0, 2, 3], True)
        add_tensor_821 = torch.ops.aten.add.Tensor(var_correction_372, 0.001)
        var_correction_372 = None
        sqrt_default_186 = torch.ops.aten.sqrt.default(add_tensor_821)
        add_tensor_821 = None
        reciprocal_default_186 = torch.ops.aten.reciprocal.default(sqrt_default_186)
        sqrt_default_186 = None
        sub_tensor_186 = torch.ops.aten.sub.Tensor(
            convolution_default_344, mean_dim_186
        )
        mul_tensor_1302 = torch.ops.aten.mul.Tensor(
            sub_tensor_186, reciprocal_default_186
        )
        sub_tensor_186 = None
        squeeze_dim_1116 = torch.ops.aten.squeeze.dim(mean_dim_186, 3)
        mean_dim_186 = None
        squeeze_dim_1117 = torch.ops.aten.squeeze.dim(squeeze_dim_1116, 2)
        squeeze_dim_1116 = None
        squeeze_dim_1118 = torch.ops.aten.squeeze.dim(squeeze_dim_1117, 0)
        squeeze_dim_1117 = None
        squeeze_dim_1119 = torch.ops.aten.squeeze.dim(reciprocal_default_186, 3)
        reciprocal_default_186 = None
        squeeze_dim_1120 = torch.ops.aten.squeeze.dim(squeeze_dim_1119, 2)
        squeeze_dim_1119 = None
        squeeze_dim_1121 = torch.ops.aten.squeeze.dim(squeeze_dim_1120, 0)
        squeeze_dim_1120 = None
        unsqueeze_default_744 = torch.ops.aten.unsqueeze.default(primals_717, -1)
        unsqueeze_default_745 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_744, -1
        )
        unsqueeze_default_744 = None
        unsqueeze_default_746 = torch.ops.aten.unsqueeze.default(primals_718, -1)
        primals_718 = None
        unsqueeze_default_747 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_746, -1
        )
        unsqueeze_default_746 = None
        mul_tensor_1308 = torch.ops.aten.mul.Tensor(
            mul_tensor_1302, unsqueeze_default_745
        )
        mul_tensor_1302 = unsqueeze_default_745 = None
        add_tensor_824 = torch.ops.aten.add.Tensor(
            mul_tensor_1308, unsqueeze_default_747
        )
        mul_tensor_1308 = unsqueeze_default_747 = None
        relu_default_186 = torch.ops.aten.relu.default(add_tensor_824)
        add_tensor_824 = None
        convolution_default_345 = torch.ops.aten.convolution.default(
            relu_default_186,
            primals_719,
            None,
            [1, 1],
            [3, 3],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_346 = torch.ops.aten.convolution.default(
            convolution_default_345,
            primals_720,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_374 = torch.ops.aten.var.correction(
            convolution_default_346, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_187 = torch.ops.aten.mean.dim(convolution_default_346, [0, 2, 3], True)
        add_tensor_825 = torch.ops.aten.add.Tensor(var_correction_374, 0.001)
        var_correction_374 = None
        sqrt_default_187 = torch.ops.aten.sqrt.default(add_tensor_825)
        add_tensor_825 = None
        reciprocal_default_187 = torch.ops.aten.reciprocal.default(sqrt_default_187)
        sqrt_default_187 = None
        sub_tensor_187 = torch.ops.aten.sub.Tensor(
            convolution_default_346, mean_dim_187
        )
        mul_tensor_1309 = torch.ops.aten.mul.Tensor(
            sub_tensor_187, reciprocal_default_187
        )
        sub_tensor_187 = None
        squeeze_dim_1122 = torch.ops.aten.squeeze.dim(mean_dim_187, 3)
        mean_dim_187 = None
        squeeze_dim_1123 = torch.ops.aten.squeeze.dim(squeeze_dim_1122, 2)
        squeeze_dim_1122 = None
        squeeze_dim_1124 = torch.ops.aten.squeeze.dim(squeeze_dim_1123, 0)
        squeeze_dim_1123 = None
        squeeze_dim_1125 = torch.ops.aten.squeeze.dim(reciprocal_default_187, 3)
        reciprocal_default_187 = None
        squeeze_dim_1126 = torch.ops.aten.squeeze.dim(squeeze_dim_1125, 2)
        squeeze_dim_1125 = None
        squeeze_dim_1127 = torch.ops.aten.squeeze.dim(squeeze_dim_1126, 0)
        squeeze_dim_1126 = None
        unsqueeze_default_748 = torch.ops.aten.unsqueeze.default(primals_721, -1)
        unsqueeze_default_749 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_748, -1
        )
        unsqueeze_default_748 = None
        unsqueeze_default_750 = torch.ops.aten.unsqueeze.default(primals_722, -1)
        primals_722 = None
        unsqueeze_default_751 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_750, -1
        )
        unsqueeze_default_750 = None
        mul_tensor_1315 = torch.ops.aten.mul.Tensor(
            mul_tensor_1309, unsqueeze_default_749
        )
        mul_tensor_1309 = unsqueeze_default_749 = None
        add_tensor_828 = torch.ops.aten.add.Tensor(
            mul_tensor_1315, unsqueeze_default_751
        )
        mul_tensor_1315 = unsqueeze_default_751 = None
        add_tensor_829 = torch.ops.aten.add.Tensor(getitem_12, add_tensor_828)
        add_tensor_828 = None
        constant_pad_nd_default_28 = torch.ops.aten.constant_pad_nd.default(
            add_tensor_803, [1, 1, 1, 1], 0.0
        )
        add_tensor_803 = None
        avg_pool2d_default_48 = torch.ops.aten.avg_pool2d.default(
            constant_pad_nd_default_28, [3, 3], [2, 2], [0, 0], False, False
        )
        constant_pad_nd_default_29 = torch.ops.aten.constant_pad_nd.default(
            relu_default_183, [2, 2, 2, 2], 0.0
        )
        convolution_default_347 = torch.ops.aten.convolution.default(
            constant_pad_nd_default_29,
            primals_16,
            None,
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_348 = torch.ops.aten.convolution.default(
            convolution_default_347,
            primals_723,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_376 = torch.ops.aten.var.correction(
            convolution_default_348, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_188 = torch.ops.aten.mean.dim(convolution_default_348, [0, 2, 3], True)
        add_tensor_830 = torch.ops.aten.add.Tensor(var_correction_376, 0.001)
        var_correction_376 = None
        sqrt_default_188 = torch.ops.aten.sqrt.default(add_tensor_830)
        add_tensor_830 = None
        reciprocal_default_188 = torch.ops.aten.reciprocal.default(sqrt_default_188)
        sqrt_default_188 = None
        sub_tensor_188 = torch.ops.aten.sub.Tensor(
            convolution_default_348, mean_dim_188
        )
        mul_tensor_1316 = torch.ops.aten.mul.Tensor(
            sub_tensor_188, reciprocal_default_188
        )
        sub_tensor_188 = None
        squeeze_dim_1128 = torch.ops.aten.squeeze.dim(mean_dim_188, 3)
        mean_dim_188 = None
        squeeze_dim_1129 = torch.ops.aten.squeeze.dim(squeeze_dim_1128, 2)
        squeeze_dim_1128 = None
        squeeze_dim_1130 = torch.ops.aten.squeeze.dim(squeeze_dim_1129, 0)
        squeeze_dim_1129 = None
        squeeze_dim_1131 = torch.ops.aten.squeeze.dim(reciprocal_default_188, 3)
        reciprocal_default_188 = None
        squeeze_dim_1132 = torch.ops.aten.squeeze.dim(squeeze_dim_1131, 2)
        squeeze_dim_1131 = None
        squeeze_dim_1133 = torch.ops.aten.squeeze.dim(squeeze_dim_1132, 0)
        squeeze_dim_1132 = None
        unsqueeze_default_752 = torch.ops.aten.unsqueeze.default(primals_724, -1)
        unsqueeze_default_753 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_752, -1
        )
        unsqueeze_default_752 = None
        unsqueeze_default_754 = torch.ops.aten.unsqueeze.default(primals_725, -1)
        primals_725 = None
        unsqueeze_default_755 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_754, -1
        )
        unsqueeze_default_754 = None
        mul_tensor_1322 = torch.ops.aten.mul.Tensor(
            mul_tensor_1316, unsqueeze_default_753
        )
        mul_tensor_1316 = unsqueeze_default_753 = None
        add_tensor_833 = torch.ops.aten.add.Tensor(
            mul_tensor_1322, unsqueeze_default_755
        )
        mul_tensor_1322 = unsqueeze_default_755 = None
        relu_default_188 = torch.ops.aten.relu.default(add_tensor_833)
        add_tensor_833 = None
        convolution_default_349 = torch.ops.aten.convolution.default(
            relu_default_188,
            primals_726,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_350 = torch.ops.aten.convolution.default(
            convolution_default_349,
            primals_727,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_378 = torch.ops.aten.var.correction(
            convolution_default_350, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_189 = torch.ops.aten.mean.dim(convolution_default_350, [0, 2, 3], True)
        add_tensor_834 = torch.ops.aten.add.Tensor(var_correction_378, 0.001)
        var_correction_378 = None
        sqrt_default_189 = torch.ops.aten.sqrt.default(add_tensor_834)
        add_tensor_834 = None
        reciprocal_default_189 = torch.ops.aten.reciprocal.default(sqrt_default_189)
        sqrt_default_189 = None
        sub_tensor_189 = torch.ops.aten.sub.Tensor(
            convolution_default_350, mean_dim_189
        )
        mul_tensor_1323 = torch.ops.aten.mul.Tensor(
            sub_tensor_189, reciprocal_default_189
        )
        sub_tensor_189 = None
        squeeze_dim_1134 = torch.ops.aten.squeeze.dim(mean_dim_189, 3)
        mean_dim_189 = None
        squeeze_dim_1135 = torch.ops.aten.squeeze.dim(squeeze_dim_1134, 2)
        squeeze_dim_1134 = None
        squeeze_dim_1136 = torch.ops.aten.squeeze.dim(squeeze_dim_1135, 0)
        squeeze_dim_1135 = None
        squeeze_dim_1137 = torch.ops.aten.squeeze.dim(reciprocal_default_189, 3)
        reciprocal_default_189 = None
        squeeze_dim_1138 = torch.ops.aten.squeeze.dim(squeeze_dim_1137, 2)
        squeeze_dim_1137 = None
        squeeze_dim_1139 = torch.ops.aten.squeeze.dim(squeeze_dim_1138, 0)
        squeeze_dim_1138 = None
        unsqueeze_default_756 = torch.ops.aten.unsqueeze.default(primals_728, -1)
        unsqueeze_default_757 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_756, -1
        )
        unsqueeze_default_756 = None
        unsqueeze_default_758 = torch.ops.aten.unsqueeze.default(primals_729, -1)
        primals_729 = None
        unsqueeze_default_759 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_758, -1
        )
        unsqueeze_default_758 = None
        mul_tensor_1329 = torch.ops.aten.mul.Tensor(
            mul_tensor_1323, unsqueeze_default_757
        )
        mul_tensor_1323 = unsqueeze_default_757 = None
        add_tensor_837 = torch.ops.aten.add.Tensor(
            mul_tensor_1329, unsqueeze_default_759
        )
        mul_tensor_1329 = unsqueeze_default_759 = None
        add_tensor_838 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_48, add_tensor_837
        )
        avg_pool2d_default_48 = add_tensor_837 = None
        avg_pool2d_default_49 = torch.ops.aten.avg_pool2d.default(
            add_tensor_820, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_839 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_49, add_tensor_829
        )
        avg_pool2d_default_49 = None
        relu_default_189 = torch.ops.aten.relu.default(add_tensor_820)
        convolution_default_351 = torch.ops.aten.convolution.default(
            relu_default_189,
            primals_730,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_189 = None
        convolution_default_352 = torch.ops.aten.convolution.default(
            convolution_default_351,
            primals_731,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_380 = torch.ops.aten.var.correction(
            convolution_default_352, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_190 = torch.ops.aten.mean.dim(convolution_default_352, [0, 2, 3], True)
        add_tensor_840 = torch.ops.aten.add.Tensor(var_correction_380, 0.001)
        var_correction_380 = None
        sqrt_default_190 = torch.ops.aten.sqrt.default(add_tensor_840)
        add_tensor_840 = None
        reciprocal_default_190 = torch.ops.aten.reciprocal.default(sqrt_default_190)
        sqrt_default_190 = None
        sub_tensor_190 = torch.ops.aten.sub.Tensor(
            convolution_default_352, mean_dim_190
        )
        mul_tensor_1330 = torch.ops.aten.mul.Tensor(
            sub_tensor_190, reciprocal_default_190
        )
        sub_tensor_190 = None
        squeeze_dim_1140 = torch.ops.aten.squeeze.dim(mean_dim_190, 3)
        mean_dim_190 = None
        squeeze_dim_1141 = torch.ops.aten.squeeze.dim(squeeze_dim_1140, 2)
        squeeze_dim_1140 = None
        squeeze_dim_1142 = torch.ops.aten.squeeze.dim(squeeze_dim_1141, 0)
        squeeze_dim_1141 = None
        squeeze_dim_1143 = torch.ops.aten.squeeze.dim(reciprocal_default_190, 3)
        reciprocal_default_190 = None
        squeeze_dim_1144 = torch.ops.aten.squeeze.dim(squeeze_dim_1143, 2)
        squeeze_dim_1143 = None
        squeeze_dim_1145 = torch.ops.aten.squeeze.dim(squeeze_dim_1144, 0)
        squeeze_dim_1144 = None
        unsqueeze_default_760 = torch.ops.aten.unsqueeze.default(primals_732, -1)
        unsqueeze_default_761 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_760, -1
        )
        unsqueeze_default_760 = None
        unsqueeze_default_762 = torch.ops.aten.unsqueeze.default(primals_733, -1)
        primals_733 = None
        unsqueeze_default_763 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_762, -1
        )
        unsqueeze_default_762 = None
        mul_tensor_1336 = torch.ops.aten.mul.Tensor(
            mul_tensor_1330, unsqueeze_default_761
        )
        mul_tensor_1330 = unsqueeze_default_761 = None
        add_tensor_843 = torch.ops.aten.add.Tensor(
            mul_tensor_1336, unsqueeze_default_763
        )
        mul_tensor_1336 = unsqueeze_default_763 = None
        relu_default_190 = torch.ops.aten.relu.default(add_tensor_843)
        add_tensor_843 = None
        convolution_default_353 = torch.ops.aten.convolution.default(
            relu_default_190,
            primals_734,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_354 = torch.ops.aten.convolution.default(
            convolution_default_353,
            primals_735,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_382 = torch.ops.aten.var.correction(
            convolution_default_354, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_191 = torch.ops.aten.mean.dim(convolution_default_354, [0, 2, 3], True)
        add_tensor_844 = torch.ops.aten.add.Tensor(var_correction_382, 0.001)
        var_correction_382 = None
        sqrt_default_191 = torch.ops.aten.sqrt.default(add_tensor_844)
        add_tensor_844 = None
        reciprocal_default_191 = torch.ops.aten.reciprocal.default(sqrt_default_191)
        sqrt_default_191 = None
        sub_tensor_191 = torch.ops.aten.sub.Tensor(
            convolution_default_354, mean_dim_191
        )
        mul_tensor_1337 = torch.ops.aten.mul.Tensor(
            sub_tensor_191, reciprocal_default_191
        )
        sub_tensor_191 = None
        squeeze_dim_1146 = torch.ops.aten.squeeze.dim(mean_dim_191, 3)
        mean_dim_191 = None
        squeeze_dim_1147 = torch.ops.aten.squeeze.dim(squeeze_dim_1146, 2)
        squeeze_dim_1146 = None
        squeeze_dim_1148 = torch.ops.aten.squeeze.dim(squeeze_dim_1147, 0)
        squeeze_dim_1147 = None
        squeeze_dim_1149 = torch.ops.aten.squeeze.dim(reciprocal_default_191, 3)
        reciprocal_default_191 = None
        squeeze_dim_1150 = torch.ops.aten.squeeze.dim(squeeze_dim_1149, 2)
        squeeze_dim_1149 = None
        squeeze_dim_1151 = torch.ops.aten.squeeze.dim(squeeze_dim_1150, 0)
        squeeze_dim_1150 = None
        unsqueeze_default_764 = torch.ops.aten.unsqueeze.default(primals_736, -1)
        unsqueeze_default_765 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_764, -1
        )
        unsqueeze_default_764 = None
        unsqueeze_default_766 = torch.ops.aten.unsqueeze.default(primals_737, -1)
        primals_737 = None
        unsqueeze_default_767 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_766, -1
        )
        unsqueeze_default_766 = None
        mul_tensor_1343 = torch.ops.aten.mul.Tensor(
            mul_tensor_1337, unsqueeze_default_765
        )
        mul_tensor_1337 = unsqueeze_default_765 = None
        add_tensor_847 = torch.ops.aten.add.Tensor(
            mul_tensor_1343, unsqueeze_default_767
        )
        mul_tensor_1343 = unsqueeze_default_767 = None
        add_tensor_848 = torch.ops.aten.add.Tensor(add_tensor_847, getitem_12)
        add_tensor_847 = getitem_12 = None
        cat_default_18 = torch.ops.aten.cat.default(
            [add_tensor_829, add_tensor_838, add_tensor_839, add_tensor_848], 1
        )
        add_tensor_829 = add_tensor_838 = add_tensor_839 = add_tensor_848 = None
        avg_pool2d_default_50 = torch.ops.aten.avg_pool2d.default(
            relu_default_168, [1, 1], [2, 2], [0, 0], False, False
        )
        convolution_default_355 = torch.ops.aten.convolution.default(
            avg_pool2d_default_50,
            primals_738,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        constant_pad_nd_default_31 = torch.ops.aten.constant_pad_nd.default(
            relu_default_168, [-1, 1, -1, 1], 0.0
        )
        avg_pool2d_default_51 = torch.ops.aten.avg_pool2d.default(
            constant_pad_nd_default_31, [1, 1], [2, 2], [0, 0], False, False
        )
        convolution_default_356 = torch.ops.aten.convolution.default(
            avg_pool2d_default_51,
            primals_739,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        cat_default_19 = torch.ops.aten.cat.default(
            [convolution_default_355, convolution_default_356], 1
        )
        convolution_default_355 = convolution_default_356 = None
        var_correction_384 = torch.ops.aten.var.correction(
            cat_default_19, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_192 = torch.ops.aten.mean.dim(cat_default_19, [0, 2, 3], True)
        add_tensor_849 = torch.ops.aten.add.Tensor(var_correction_384, 0.001)
        var_correction_384 = None
        sqrt_default_192 = torch.ops.aten.sqrt.default(add_tensor_849)
        add_tensor_849 = None
        reciprocal_default_192 = torch.ops.aten.reciprocal.default(sqrt_default_192)
        sqrt_default_192 = None
        sub_tensor_192 = torch.ops.aten.sub.Tensor(cat_default_19, mean_dim_192)
        mul_tensor_1344 = torch.ops.aten.mul.Tensor(
            sub_tensor_192, reciprocal_default_192
        )
        sub_tensor_192 = None
        unsqueeze_default_768 = torch.ops.aten.unsqueeze.default(primals_740, -1)
        unsqueeze_default_769 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_768, -1
        )
        unsqueeze_default_768 = None
        unsqueeze_default_770 = torch.ops.aten.unsqueeze.default(primals_741, -1)
        unsqueeze_default_771 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_770, -1
        )
        unsqueeze_default_770 = None
        mul_tensor_1350 = torch.ops.aten.mul.Tensor(
            mul_tensor_1344, unsqueeze_default_769
        )
        mul_tensor_1344 = unsqueeze_default_769 = None
        add_tensor_852 = torch.ops.aten.add.Tensor(
            mul_tensor_1350, unsqueeze_default_771
        )
        mul_tensor_1350 = unsqueeze_default_771 = None
        relu_default_192 = torch.ops.aten.relu.default(cat_default_18)
        cat_default_18 = None
        convolution_default_357 = torch.ops.aten.convolution.default(
            relu_default_192,
            primals_742,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_386 = torch.ops.aten.var.correction(
            convolution_default_357, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_193 = torch.ops.aten.mean.dim(convolution_default_357, [0, 2, 3], True)
        add_tensor_853 = torch.ops.aten.add.Tensor(var_correction_386, 0.001)
        var_correction_386 = None
        sqrt_default_193 = torch.ops.aten.sqrt.default(add_tensor_853)
        add_tensor_853 = None
        reciprocal_default_193 = torch.ops.aten.reciprocal.default(sqrt_default_193)
        sqrt_default_193 = None
        sub_tensor_193 = torch.ops.aten.sub.Tensor(
            convolution_default_357, mean_dim_193
        )
        mul_tensor_1351 = torch.ops.aten.mul.Tensor(
            sub_tensor_193, reciprocal_default_193
        )
        sub_tensor_193 = None
        unsqueeze_default_772 = torch.ops.aten.unsqueeze.default(primals_743, -1)
        unsqueeze_default_773 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_772, -1
        )
        unsqueeze_default_772 = None
        unsqueeze_default_774 = torch.ops.aten.unsqueeze.default(primals_744, -1)
        unsqueeze_default_775 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_774, -1
        )
        unsqueeze_default_774 = None
        mul_tensor_1357 = torch.ops.aten.mul.Tensor(
            mul_tensor_1351, unsqueeze_default_773
        )
        mul_tensor_1351 = unsqueeze_default_773 = None
        add_tensor_856 = torch.ops.aten.add.Tensor(
            mul_tensor_1357, unsqueeze_default_775
        )
        mul_tensor_1357 = unsqueeze_default_775 = None
        relu_default_193 = torch.ops.aten.relu.default(add_tensor_856)
        convolution_default_358 = torch.ops.aten.convolution.default(
            relu_default_193,
            primals_745,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_359 = torch.ops.aten.convolution.default(
            convolution_default_358,
            primals_746,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_388 = torch.ops.aten.var.correction(
            convolution_default_359, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_194 = torch.ops.aten.mean.dim(convolution_default_359, [0, 2, 3], True)
        add_tensor_857 = torch.ops.aten.add.Tensor(var_correction_388, 0.001)
        var_correction_388 = None
        sqrt_default_194 = torch.ops.aten.sqrt.default(add_tensor_857)
        add_tensor_857 = None
        reciprocal_default_194 = torch.ops.aten.reciprocal.default(sqrt_default_194)
        sqrt_default_194 = None
        sub_tensor_194 = torch.ops.aten.sub.Tensor(
            convolution_default_359, mean_dim_194
        )
        mul_tensor_1358 = torch.ops.aten.mul.Tensor(
            sub_tensor_194, reciprocal_default_194
        )
        sub_tensor_194 = None
        squeeze_dim_1164 = torch.ops.aten.squeeze.dim(mean_dim_194, 3)
        mean_dim_194 = None
        squeeze_dim_1165 = torch.ops.aten.squeeze.dim(squeeze_dim_1164, 2)
        squeeze_dim_1164 = None
        squeeze_dim_1166 = torch.ops.aten.squeeze.dim(squeeze_dim_1165, 0)
        squeeze_dim_1165 = None
        squeeze_dim_1167 = torch.ops.aten.squeeze.dim(reciprocal_default_194, 3)
        reciprocal_default_194 = None
        squeeze_dim_1168 = torch.ops.aten.squeeze.dim(squeeze_dim_1167, 2)
        squeeze_dim_1167 = None
        squeeze_dim_1169 = torch.ops.aten.squeeze.dim(squeeze_dim_1168, 0)
        squeeze_dim_1168 = None
        unsqueeze_default_776 = torch.ops.aten.unsqueeze.default(primals_747, -1)
        unsqueeze_default_777 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_776, -1
        )
        unsqueeze_default_776 = None
        unsqueeze_default_778 = torch.ops.aten.unsqueeze.default(primals_748, -1)
        primals_748 = None
        unsqueeze_default_779 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_778, -1
        )
        unsqueeze_default_778 = None
        mul_tensor_1364 = torch.ops.aten.mul.Tensor(
            mul_tensor_1358, unsqueeze_default_777
        )
        mul_tensor_1358 = unsqueeze_default_777 = None
        add_tensor_860 = torch.ops.aten.add.Tensor(
            mul_tensor_1364, unsqueeze_default_779
        )
        mul_tensor_1364 = unsqueeze_default_779 = None
        relu_default_194 = torch.ops.aten.relu.default(add_tensor_860)
        add_tensor_860 = None
        convolution_default_360 = torch.ops.aten.convolution.default(
            relu_default_194,
            primals_749,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_361 = torch.ops.aten.convolution.default(
            convolution_default_360,
            primals_750,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_390 = torch.ops.aten.var.correction(
            convolution_default_361, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_195 = torch.ops.aten.mean.dim(convolution_default_361, [0, 2, 3], True)
        add_tensor_861 = torch.ops.aten.add.Tensor(var_correction_390, 0.001)
        var_correction_390 = None
        sqrt_default_195 = torch.ops.aten.sqrt.default(add_tensor_861)
        add_tensor_861 = None
        reciprocal_default_195 = torch.ops.aten.reciprocal.default(sqrt_default_195)
        sqrt_default_195 = None
        sub_tensor_195 = torch.ops.aten.sub.Tensor(
            convolution_default_361, mean_dim_195
        )
        mul_tensor_1365 = torch.ops.aten.mul.Tensor(
            sub_tensor_195, reciprocal_default_195
        )
        sub_tensor_195 = None
        squeeze_dim_1170 = torch.ops.aten.squeeze.dim(mean_dim_195, 3)
        mean_dim_195 = None
        squeeze_dim_1171 = torch.ops.aten.squeeze.dim(squeeze_dim_1170, 2)
        squeeze_dim_1170 = None
        squeeze_dim_1172 = torch.ops.aten.squeeze.dim(squeeze_dim_1171, 0)
        squeeze_dim_1171 = None
        squeeze_dim_1173 = torch.ops.aten.squeeze.dim(reciprocal_default_195, 3)
        reciprocal_default_195 = None
        squeeze_dim_1174 = torch.ops.aten.squeeze.dim(squeeze_dim_1173, 2)
        squeeze_dim_1173 = None
        squeeze_dim_1175 = torch.ops.aten.squeeze.dim(squeeze_dim_1174, 0)
        squeeze_dim_1174 = None
        unsqueeze_default_780 = torch.ops.aten.unsqueeze.default(primals_751, -1)
        unsqueeze_default_781 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_780, -1
        )
        unsqueeze_default_780 = None
        unsqueeze_default_782 = torch.ops.aten.unsqueeze.default(primals_752, -1)
        primals_752 = None
        unsqueeze_default_783 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_782, -1
        )
        unsqueeze_default_782 = None
        mul_tensor_1371 = torch.ops.aten.mul.Tensor(
            mul_tensor_1365, unsqueeze_default_781
        )
        mul_tensor_1365 = unsqueeze_default_781 = None
        add_tensor_864 = torch.ops.aten.add.Tensor(
            mul_tensor_1371, unsqueeze_default_783
        )
        mul_tensor_1371 = unsqueeze_default_783 = None
        relu_default_195 = torch.ops.aten.relu.default(add_tensor_852)
        convolution_default_362 = torch.ops.aten.convolution.default(
            relu_default_195,
            primals_753,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_363 = torch.ops.aten.convolution.default(
            convolution_default_362,
            primals_754,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_392 = torch.ops.aten.var.correction(
            convolution_default_363, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_196 = torch.ops.aten.mean.dim(convolution_default_363, [0, 2, 3], True)
        add_tensor_865 = torch.ops.aten.add.Tensor(var_correction_392, 0.001)
        var_correction_392 = None
        sqrt_default_196 = torch.ops.aten.sqrt.default(add_tensor_865)
        add_tensor_865 = None
        reciprocal_default_196 = torch.ops.aten.reciprocal.default(sqrt_default_196)
        sqrt_default_196 = None
        sub_tensor_196 = torch.ops.aten.sub.Tensor(
            convolution_default_363, mean_dim_196
        )
        mul_tensor_1372 = torch.ops.aten.mul.Tensor(
            sub_tensor_196, reciprocal_default_196
        )
        sub_tensor_196 = None
        squeeze_dim_1176 = torch.ops.aten.squeeze.dim(mean_dim_196, 3)
        mean_dim_196 = None
        squeeze_dim_1177 = torch.ops.aten.squeeze.dim(squeeze_dim_1176, 2)
        squeeze_dim_1176 = None
        squeeze_dim_1178 = torch.ops.aten.squeeze.dim(squeeze_dim_1177, 0)
        squeeze_dim_1177 = None
        squeeze_dim_1179 = torch.ops.aten.squeeze.dim(reciprocal_default_196, 3)
        reciprocal_default_196 = None
        squeeze_dim_1180 = torch.ops.aten.squeeze.dim(squeeze_dim_1179, 2)
        squeeze_dim_1179 = None
        squeeze_dim_1181 = torch.ops.aten.squeeze.dim(squeeze_dim_1180, 0)
        squeeze_dim_1180 = None
        unsqueeze_default_784 = torch.ops.aten.unsqueeze.default(primals_755, -1)
        unsqueeze_default_785 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_784, -1
        )
        unsqueeze_default_784 = None
        unsqueeze_default_786 = torch.ops.aten.unsqueeze.default(primals_756, -1)
        primals_756 = None
        unsqueeze_default_787 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_786, -1
        )
        unsqueeze_default_786 = None
        mul_tensor_1378 = torch.ops.aten.mul.Tensor(
            mul_tensor_1372, unsqueeze_default_785
        )
        mul_tensor_1372 = unsqueeze_default_785 = None
        add_tensor_868 = torch.ops.aten.add.Tensor(
            mul_tensor_1378, unsqueeze_default_787
        )
        mul_tensor_1378 = unsqueeze_default_787 = None
        relu_default_196 = torch.ops.aten.relu.default(add_tensor_868)
        add_tensor_868 = None
        convolution_default_364 = torch.ops.aten.convolution.default(
            relu_default_196,
            primals_757,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_365 = torch.ops.aten.convolution.default(
            convolution_default_364,
            primals_758,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_394 = torch.ops.aten.var.correction(
            convolution_default_365, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_197 = torch.ops.aten.mean.dim(convolution_default_365, [0, 2, 3], True)
        add_tensor_869 = torch.ops.aten.add.Tensor(var_correction_394, 0.001)
        var_correction_394 = None
        sqrt_default_197 = torch.ops.aten.sqrt.default(add_tensor_869)
        add_tensor_869 = None
        reciprocal_default_197 = torch.ops.aten.reciprocal.default(sqrt_default_197)
        sqrt_default_197 = None
        sub_tensor_197 = torch.ops.aten.sub.Tensor(
            convolution_default_365, mean_dim_197
        )
        mul_tensor_1379 = torch.ops.aten.mul.Tensor(
            sub_tensor_197, reciprocal_default_197
        )
        sub_tensor_197 = None
        squeeze_dim_1182 = torch.ops.aten.squeeze.dim(mean_dim_197, 3)
        mean_dim_197 = None
        squeeze_dim_1183 = torch.ops.aten.squeeze.dim(squeeze_dim_1182, 2)
        squeeze_dim_1182 = None
        squeeze_dim_1184 = torch.ops.aten.squeeze.dim(squeeze_dim_1183, 0)
        squeeze_dim_1183 = None
        squeeze_dim_1185 = torch.ops.aten.squeeze.dim(reciprocal_default_197, 3)
        reciprocal_default_197 = None
        squeeze_dim_1186 = torch.ops.aten.squeeze.dim(squeeze_dim_1185, 2)
        squeeze_dim_1185 = None
        squeeze_dim_1187 = torch.ops.aten.squeeze.dim(squeeze_dim_1186, 0)
        squeeze_dim_1186 = None
        unsqueeze_default_788 = torch.ops.aten.unsqueeze.default(primals_759, -1)
        unsqueeze_default_789 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_788, -1
        )
        unsqueeze_default_788 = None
        unsqueeze_default_790 = torch.ops.aten.unsqueeze.default(primals_760, -1)
        primals_760 = None
        unsqueeze_default_791 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_790, -1
        )
        unsqueeze_default_790 = None
        mul_tensor_1385 = torch.ops.aten.mul.Tensor(
            mul_tensor_1379, unsqueeze_default_789
        )
        mul_tensor_1379 = unsqueeze_default_789 = None
        add_tensor_872 = torch.ops.aten.add.Tensor(
            mul_tensor_1385, unsqueeze_default_791
        )
        mul_tensor_1385 = unsqueeze_default_791 = None
        add_tensor_873 = torch.ops.aten.add.Tensor(add_tensor_864, add_tensor_872)
        add_tensor_864 = add_tensor_872 = None
        convolution_default_366 = torch.ops.aten.convolution.default(
            relu_default_195,
            primals_761,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_367 = torch.ops.aten.convolution.default(
            convolution_default_366,
            primals_762,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_396 = torch.ops.aten.var.correction(
            convolution_default_367, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_198 = torch.ops.aten.mean.dim(convolution_default_367, [0, 2, 3], True)
        add_tensor_874 = torch.ops.aten.add.Tensor(var_correction_396, 0.001)
        var_correction_396 = None
        sqrt_default_198 = torch.ops.aten.sqrt.default(add_tensor_874)
        add_tensor_874 = None
        reciprocal_default_198 = torch.ops.aten.reciprocal.default(sqrt_default_198)
        sqrt_default_198 = None
        sub_tensor_198 = torch.ops.aten.sub.Tensor(
            convolution_default_367, mean_dim_198
        )
        mul_tensor_1386 = torch.ops.aten.mul.Tensor(
            sub_tensor_198, reciprocal_default_198
        )
        sub_tensor_198 = None
        squeeze_dim_1188 = torch.ops.aten.squeeze.dim(mean_dim_198, 3)
        mean_dim_198 = None
        squeeze_dim_1189 = torch.ops.aten.squeeze.dim(squeeze_dim_1188, 2)
        squeeze_dim_1188 = None
        squeeze_dim_1190 = torch.ops.aten.squeeze.dim(squeeze_dim_1189, 0)
        squeeze_dim_1189 = None
        squeeze_dim_1191 = torch.ops.aten.squeeze.dim(reciprocal_default_198, 3)
        reciprocal_default_198 = None
        squeeze_dim_1192 = torch.ops.aten.squeeze.dim(squeeze_dim_1191, 2)
        squeeze_dim_1191 = None
        squeeze_dim_1193 = torch.ops.aten.squeeze.dim(squeeze_dim_1192, 0)
        squeeze_dim_1192 = None
        unsqueeze_default_792 = torch.ops.aten.unsqueeze.default(primals_763, -1)
        unsqueeze_default_793 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_792, -1
        )
        unsqueeze_default_792 = None
        unsqueeze_default_794 = torch.ops.aten.unsqueeze.default(primals_764, -1)
        primals_764 = None
        unsqueeze_default_795 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_794, -1
        )
        unsqueeze_default_794 = None
        mul_tensor_1392 = torch.ops.aten.mul.Tensor(
            mul_tensor_1386, unsqueeze_default_793
        )
        mul_tensor_1386 = unsqueeze_default_793 = None
        add_tensor_877 = torch.ops.aten.add.Tensor(
            mul_tensor_1392, unsqueeze_default_795
        )
        mul_tensor_1392 = unsqueeze_default_795 = None
        relu_default_198 = torch.ops.aten.relu.default(add_tensor_877)
        add_tensor_877 = None
        convolution_default_368 = torch.ops.aten.convolution.default(
            relu_default_198,
            primals_765,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_369 = torch.ops.aten.convolution.default(
            convolution_default_368,
            primals_766,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_398 = torch.ops.aten.var.correction(
            convolution_default_369, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_199 = torch.ops.aten.mean.dim(convolution_default_369, [0, 2, 3], True)
        add_tensor_878 = torch.ops.aten.add.Tensor(var_correction_398, 0.001)
        var_correction_398 = None
        sqrt_default_199 = torch.ops.aten.sqrt.default(add_tensor_878)
        add_tensor_878 = None
        reciprocal_default_199 = torch.ops.aten.reciprocal.default(sqrt_default_199)
        sqrt_default_199 = None
        sub_tensor_199 = torch.ops.aten.sub.Tensor(
            convolution_default_369, mean_dim_199
        )
        mul_tensor_1393 = torch.ops.aten.mul.Tensor(
            sub_tensor_199, reciprocal_default_199
        )
        sub_tensor_199 = None
        squeeze_dim_1194 = torch.ops.aten.squeeze.dim(mean_dim_199, 3)
        mean_dim_199 = None
        squeeze_dim_1195 = torch.ops.aten.squeeze.dim(squeeze_dim_1194, 2)
        squeeze_dim_1194 = None
        squeeze_dim_1196 = torch.ops.aten.squeeze.dim(squeeze_dim_1195, 0)
        squeeze_dim_1195 = None
        squeeze_dim_1197 = torch.ops.aten.squeeze.dim(reciprocal_default_199, 3)
        reciprocal_default_199 = None
        squeeze_dim_1198 = torch.ops.aten.squeeze.dim(squeeze_dim_1197, 2)
        squeeze_dim_1197 = None
        squeeze_dim_1199 = torch.ops.aten.squeeze.dim(squeeze_dim_1198, 0)
        squeeze_dim_1198 = None
        unsqueeze_default_796 = torch.ops.aten.unsqueeze.default(primals_767, -1)
        unsqueeze_default_797 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_796, -1
        )
        unsqueeze_default_796 = None
        unsqueeze_default_798 = torch.ops.aten.unsqueeze.default(primals_768, -1)
        primals_768 = None
        unsqueeze_default_799 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_798, -1
        )
        unsqueeze_default_798 = None
        mul_tensor_1399 = torch.ops.aten.mul.Tensor(
            mul_tensor_1393, unsqueeze_default_797
        )
        mul_tensor_1393 = unsqueeze_default_797 = None
        add_tensor_881 = torch.ops.aten.add.Tensor(
            mul_tensor_1399, unsqueeze_default_799
        )
        mul_tensor_1399 = unsqueeze_default_799 = None
        convolution_default_370 = torch.ops.aten.convolution.default(
            relu_default_195,
            primals_769,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_195 = None
        convolution_default_371 = torch.ops.aten.convolution.default(
            convolution_default_370,
            primals_770,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_400 = torch.ops.aten.var.correction(
            convolution_default_371, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_200 = torch.ops.aten.mean.dim(convolution_default_371, [0, 2, 3], True)
        add_tensor_882 = torch.ops.aten.add.Tensor(var_correction_400, 0.001)
        var_correction_400 = None
        sqrt_default_200 = torch.ops.aten.sqrt.default(add_tensor_882)
        add_tensor_882 = None
        reciprocal_default_200 = torch.ops.aten.reciprocal.default(sqrt_default_200)
        sqrt_default_200 = None
        sub_tensor_200 = torch.ops.aten.sub.Tensor(
            convolution_default_371, mean_dim_200
        )
        mul_tensor_1400 = torch.ops.aten.mul.Tensor(
            sub_tensor_200, reciprocal_default_200
        )
        sub_tensor_200 = None
        squeeze_dim_1200 = torch.ops.aten.squeeze.dim(mean_dim_200, 3)
        mean_dim_200 = None
        squeeze_dim_1201 = torch.ops.aten.squeeze.dim(squeeze_dim_1200, 2)
        squeeze_dim_1200 = None
        squeeze_dim_1202 = torch.ops.aten.squeeze.dim(squeeze_dim_1201, 0)
        squeeze_dim_1201 = None
        squeeze_dim_1203 = torch.ops.aten.squeeze.dim(reciprocal_default_200, 3)
        reciprocal_default_200 = None
        squeeze_dim_1204 = torch.ops.aten.squeeze.dim(squeeze_dim_1203, 2)
        squeeze_dim_1203 = None
        squeeze_dim_1205 = torch.ops.aten.squeeze.dim(squeeze_dim_1204, 0)
        squeeze_dim_1204 = None
        unsqueeze_default_800 = torch.ops.aten.unsqueeze.default(primals_771, -1)
        unsqueeze_default_801 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_800, -1
        )
        unsqueeze_default_800 = None
        unsqueeze_default_802 = torch.ops.aten.unsqueeze.default(primals_772, -1)
        primals_772 = None
        unsqueeze_default_803 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_802, -1
        )
        unsqueeze_default_802 = None
        mul_tensor_1406 = torch.ops.aten.mul.Tensor(
            mul_tensor_1400, unsqueeze_default_801
        )
        mul_tensor_1400 = unsqueeze_default_801 = None
        add_tensor_885 = torch.ops.aten.add.Tensor(
            mul_tensor_1406, unsqueeze_default_803
        )
        mul_tensor_1406 = unsqueeze_default_803 = None
        relu_default_200 = torch.ops.aten.relu.default(add_tensor_885)
        add_tensor_885 = None
        convolution_default_372 = torch.ops.aten.convolution.default(
            relu_default_200,
            primals_773,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_373 = torch.ops.aten.convolution.default(
            convolution_default_372,
            primals_774,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_402 = torch.ops.aten.var.correction(
            convolution_default_373, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_201 = torch.ops.aten.mean.dim(convolution_default_373, [0, 2, 3], True)
        add_tensor_886 = torch.ops.aten.add.Tensor(var_correction_402, 0.001)
        var_correction_402 = None
        sqrt_default_201 = torch.ops.aten.sqrt.default(add_tensor_886)
        add_tensor_886 = None
        reciprocal_default_201 = torch.ops.aten.reciprocal.default(sqrt_default_201)
        sqrt_default_201 = None
        sub_tensor_201 = torch.ops.aten.sub.Tensor(
            convolution_default_373, mean_dim_201
        )
        mul_tensor_1407 = torch.ops.aten.mul.Tensor(
            sub_tensor_201, reciprocal_default_201
        )
        sub_tensor_201 = None
        squeeze_dim_1206 = torch.ops.aten.squeeze.dim(mean_dim_201, 3)
        mean_dim_201 = None
        squeeze_dim_1207 = torch.ops.aten.squeeze.dim(squeeze_dim_1206, 2)
        squeeze_dim_1206 = None
        squeeze_dim_1208 = torch.ops.aten.squeeze.dim(squeeze_dim_1207, 0)
        squeeze_dim_1207 = None
        squeeze_dim_1209 = torch.ops.aten.squeeze.dim(reciprocal_default_201, 3)
        reciprocal_default_201 = None
        squeeze_dim_1210 = torch.ops.aten.squeeze.dim(squeeze_dim_1209, 2)
        squeeze_dim_1209 = None
        squeeze_dim_1211 = torch.ops.aten.squeeze.dim(squeeze_dim_1210, 0)
        squeeze_dim_1210 = None
        unsqueeze_default_804 = torch.ops.aten.unsqueeze.default(primals_775, -1)
        unsqueeze_default_805 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_804, -1
        )
        unsqueeze_default_804 = None
        unsqueeze_default_806 = torch.ops.aten.unsqueeze.default(primals_776, -1)
        primals_776 = None
        unsqueeze_default_807 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_806, -1
        )
        unsqueeze_default_806 = None
        mul_tensor_1413 = torch.ops.aten.mul.Tensor(
            mul_tensor_1407, unsqueeze_default_805
        )
        mul_tensor_1407 = unsqueeze_default_805 = None
        add_tensor_889 = torch.ops.aten.add.Tensor(
            mul_tensor_1413, unsqueeze_default_807
        )
        mul_tensor_1413 = unsqueeze_default_807 = None
        add_tensor_890 = torch.ops.aten.add.Tensor(add_tensor_881, add_tensor_889)
        add_tensor_881 = add_tensor_889 = None
        avg_pool2d_default_52 = torch.ops.aten.avg_pool2d.default(
            add_tensor_856, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_891 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_52, add_tensor_852
        )
        avg_pool2d_default_52 = None
        avg_pool2d_default_53 = torch.ops.aten.avg_pool2d.default(
            add_tensor_852, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_892 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_53, avg_pool2d_default_53
        )
        avg_pool2d_default_53 = None
        convolution_default_374 = torch.ops.aten.convolution.default(
            relu_default_193,
            primals_777,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_193 = None
        convolution_default_375 = torch.ops.aten.convolution.default(
            convolution_default_374,
            primals_778,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_404 = torch.ops.aten.var.correction(
            convolution_default_375, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_202 = torch.ops.aten.mean.dim(convolution_default_375, [0, 2, 3], True)
        add_tensor_893 = torch.ops.aten.add.Tensor(var_correction_404, 0.001)
        var_correction_404 = None
        sqrt_default_202 = torch.ops.aten.sqrt.default(add_tensor_893)
        add_tensor_893 = None
        reciprocal_default_202 = torch.ops.aten.reciprocal.default(sqrt_default_202)
        sqrt_default_202 = None
        sub_tensor_202 = torch.ops.aten.sub.Tensor(
            convolution_default_375, mean_dim_202
        )
        mul_tensor_1414 = torch.ops.aten.mul.Tensor(
            sub_tensor_202, reciprocal_default_202
        )
        sub_tensor_202 = None
        squeeze_dim_1212 = torch.ops.aten.squeeze.dim(mean_dim_202, 3)
        mean_dim_202 = None
        squeeze_dim_1213 = torch.ops.aten.squeeze.dim(squeeze_dim_1212, 2)
        squeeze_dim_1212 = None
        squeeze_dim_1214 = torch.ops.aten.squeeze.dim(squeeze_dim_1213, 0)
        squeeze_dim_1213 = None
        squeeze_dim_1215 = torch.ops.aten.squeeze.dim(reciprocal_default_202, 3)
        reciprocal_default_202 = None
        squeeze_dim_1216 = torch.ops.aten.squeeze.dim(squeeze_dim_1215, 2)
        squeeze_dim_1215 = None
        squeeze_dim_1217 = torch.ops.aten.squeeze.dim(squeeze_dim_1216, 0)
        squeeze_dim_1216 = None
        unsqueeze_default_808 = torch.ops.aten.unsqueeze.default(primals_779, -1)
        unsqueeze_default_809 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_808, -1
        )
        unsqueeze_default_808 = None
        unsqueeze_default_810 = torch.ops.aten.unsqueeze.default(primals_780, -1)
        primals_780 = None
        unsqueeze_default_811 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_810, -1
        )
        unsqueeze_default_810 = None
        mul_tensor_1420 = torch.ops.aten.mul.Tensor(
            mul_tensor_1414, unsqueeze_default_809
        )
        mul_tensor_1414 = unsqueeze_default_809 = None
        add_tensor_896 = torch.ops.aten.add.Tensor(
            mul_tensor_1420, unsqueeze_default_811
        )
        mul_tensor_1420 = unsqueeze_default_811 = None
        relu_default_202 = torch.ops.aten.relu.default(add_tensor_896)
        add_tensor_896 = None
        convolution_default_376 = torch.ops.aten.convolution.default(
            relu_default_202,
            primals_781,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_377 = torch.ops.aten.convolution.default(
            convolution_default_376,
            primals_782,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_406 = torch.ops.aten.var.correction(
            convolution_default_377, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_203 = torch.ops.aten.mean.dim(convolution_default_377, [0, 2, 3], True)
        add_tensor_897 = torch.ops.aten.add.Tensor(var_correction_406, 0.001)
        var_correction_406 = None
        sqrt_default_203 = torch.ops.aten.sqrt.default(add_tensor_897)
        add_tensor_897 = None
        reciprocal_default_203 = torch.ops.aten.reciprocal.default(sqrt_default_203)
        sqrt_default_203 = None
        sub_tensor_203 = torch.ops.aten.sub.Tensor(
            convolution_default_377, mean_dim_203
        )
        mul_tensor_1421 = torch.ops.aten.mul.Tensor(
            sub_tensor_203, reciprocal_default_203
        )
        sub_tensor_203 = None
        squeeze_dim_1218 = torch.ops.aten.squeeze.dim(mean_dim_203, 3)
        mean_dim_203 = None
        squeeze_dim_1219 = torch.ops.aten.squeeze.dim(squeeze_dim_1218, 2)
        squeeze_dim_1218 = None
        squeeze_dim_1220 = torch.ops.aten.squeeze.dim(squeeze_dim_1219, 0)
        squeeze_dim_1219 = None
        squeeze_dim_1221 = torch.ops.aten.squeeze.dim(reciprocal_default_203, 3)
        reciprocal_default_203 = None
        squeeze_dim_1222 = torch.ops.aten.squeeze.dim(squeeze_dim_1221, 2)
        squeeze_dim_1221 = None
        squeeze_dim_1223 = torch.ops.aten.squeeze.dim(squeeze_dim_1222, 0)
        squeeze_dim_1222 = None
        unsqueeze_default_812 = torch.ops.aten.unsqueeze.default(primals_783, -1)
        unsqueeze_default_813 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_812, -1
        )
        unsqueeze_default_812 = None
        unsqueeze_default_814 = torch.ops.aten.unsqueeze.default(primals_784, -1)
        primals_784 = None
        unsqueeze_default_815 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_814, -1
        )
        unsqueeze_default_814 = None
        mul_tensor_1427 = torch.ops.aten.mul.Tensor(
            mul_tensor_1421, unsqueeze_default_813
        )
        mul_tensor_1421 = unsqueeze_default_813 = None
        add_tensor_900 = torch.ops.aten.add.Tensor(
            mul_tensor_1427, unsqueeze_default_815
        )
        mul_tensor_1427 = unsqueeze_default_815 = None
        add_tensor_901 = torch.ops.aten.add.Tensor(add_tensor_900, add_tensor_856)
        add_tensor_900 = add_tensor_856 = None
        cat_default_20 = torch.ops.aten.cat.default(
            [
                add_tensor_852,
                add_tensor_873,
                add_tensor_890,
                add_tensor_891,
                add_tensor_892,
                add_tensor_901,
            ],
            1,
        )
        add_tensor_852 = (
            add_tensor_873
        ) = add_tensor_890 = add_tensor_891 = add_tensor_892 = add_tensor_901 = None
        convolution_default_378 = torch.ops.aten.convolution.default(
            relu_default_192,
            primals_785,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_408 = torch.ops.aten.var.correction(
            convolution_default_378, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_204 = torch.ops.aten.mean.dim(convolution_default_378, [0, 2, 3], True)
        add_tensor_902 = torch.ops.aten.add.Tensor(var_correction_408, 0.001)
        var_correction_408 = None
        sqrt_default_204 = torch.ops.aten.sqrt.default(add_tensor_902)
        add_tensor_902 = None
        reciprocal_default_204 = torch.ops.aten.reciprocal.default(sqrt_default_204)
        sqrt_default_204 = None
        sub_tensor_204 = torch.ops.aten.sub.Tensor(
            convolution_default_378, mean_dim_204
        )
        mul_tensor_1428 = torch.ops.aten.mul.Tensor(
            sub_tensor_204, reciprocal_default_204
        )
        sub_tensor_204 = None
        unsqueeze_default_816 = torch.ops.aten.unsqueeze.default(primals_786, -1)
        unsqueeze_default_817 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_816, -1
        )
        unsqueeze_default_816 = None
        unsqueeze_default_818 = torch.ops.aten.unsqueeze.default(primals_787, -1)
        unsqueeze_default_819 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_818, -1
        )
        unsqueeze_default_818 = None
        mul_tensor_1434 = torch.ops.aten.mul.Tensor(
            mul_tensor_1428, unsqueeze_default_817
        )
        mul_tensor_1428 = unsqueeze_default_817 = None
        add_tensor_905 = torch.ops.aten.add.Tensor(
            mul_tensor_1434, unsqueeze_default_819
        )
        mul_tensor_1434 = unsqueeze_default_819 = None
        relu_default_204 = torch.ops.aten.relu.default(cat_default_20)
        cat_default_20 = None
        convolution_default_379 = torch.ops.aten.convolution.default(
            relu_default_204,
            primals_788,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_410 = torch.ops.aten.var.correction(
            convolution_default_379, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_205 = torch.ops.aten.mean.dim(convolution_default_379, [0, 2, 3], True)
        add_tensor_906 = torch.ops.aten.add.Tensor(var_correction_410, 0.001)
        var_correction_410 = None
        sqrt_default_205 = torch.ops.aten.sqrt.default(add_tensor_906)
        add_tensor_906 = None
        reciprocal_default_205 = torch.ops.aten.reciprocal.default(sqrt_default_205)
        sqrt_default_205 = None
        sub_tensor_205 = torch.ops.aten.sub.Tensor(
            convolution_default_379, mean_dim_205
        )
        mul_tensor_1435 = torch.ops.aten.mul.Tensor(
            sub_tensor_205, reciprocal_default_205
        )
        sub_tensor_205 = None
        unsqueeze_default_820 = torch.ops.aten.unsqueeze.default(primals_789, -1)
        unsqueeze_default_821 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_820, -1
        )
        unsqueeze_default_820 = None
        unsqueeze_default_822 = torch.ops.aten.unsqueeze.default(primals_790, -1)
        unsqueeze_default_823 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_822, -1
        )
        unsqueeze_default_822 = None
        mul_tensor_1441 = torch.ops.aten.mul.Tensor(
            mul_tensor_1435, unsqueeze_default_821
        )
        mul_tensor_1435 = unsqueeze_default_821 = None
        add_tensor_909 = torch.ops.aten.add.Tensor(
            mul_tensor_1441, unsqueeze_default_823
        )
        mul_tensor_1441 = unsqueeze_default_823 = None
        relu_default_205 = torch.ops.aten.relu.default(add_tensor_909)
        convolution_default_380 = torch.ops.aten.convolution.default(
            relu_default_205,
            primals_791,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_381 = torch.ops.aten.convolution.default(
            convolution_default_380,
            primals_792,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_412 = torch.ops.aten.var.correction(
            convolution_default_381, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_206 = torch.ops.aten.mean.dim(convolution_default_381, [0, 2, 3], True)
        add_tensor_910 = torch.ops.aten.add.Tensor(var_correction_412, 0.001)
        var_correction_412 = None
        sqrt_default_206 = torch.ops.aten.sqrt.default(add_tensor_910)
        add_tensor_910 = None
        reciprocal_default_206 = torch.ops.aten.reciprocal.default(sqrt_default_206)
        sqrt_default_206 = None
        sub_tensor_206 = torch.ops.aten.sub.Tensor(
            convolution_default_381, mean_dim_206
        )
        mul_tensor_1442 = torch.ops.aten.mul.Tensor(
            sub_tensor_206, reciprocal_default_206
        )
        sub_tensor_206 = None
        squeeze_dim_1236 = torch.ops.aten.squeeze.dim(mean_dim_206, 3)
        mean_dim_206 = None
        squeeze_dim_1237 = torch.ops.aten.squeeze.dim(squeeze_dim_1236, 2)
        squeeze_dim_1236 = None
        squeeze_dim_1238 = torch.ops.aten.squeeze.dim(squeeze_dim_1237, 0)
        squeeze_dim_1237 = None
        squeeze_dim_1239 = torch.ops.aten.squeeze.dim(reciprocal_default_206, 3)
        reciprocal_default_206 = None
        squeeze_dim_1240 = torch.ops.aten.squeeze.dim(squeeze_dim_1239, 2)
        squeeze_dim_1239 = None
        squeeze_dim_1241 = torch.ops.aten.squeeze.dim(squeeze_dim_1240, 0)
        squeeze_dim_1240 = None
        unsqueeze_default_824 = torch.ops.aten.unsqueeze.default(primals_793, -1)
        unsqueeze_default_825 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_824, -1
        )
        unsqueeze_default_824 = None
        unsqueeze_default_826 = torch.ops.aten.unsqueeze.default(primals_794, -1)
        primals_794 = None
        unsqueeze_default_827 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_826, -1
        )
        unsqueeze_default_826 = None
        mul_tensor_1448 = torch.ops.aten.mul.Tensor(
            mul_tensor_1442, unsqueeze_default_825
        )
        mul_tensor_1442 = unsqueeze_default_825 = None
        add_tensor_913 = torch.ops.aten.add.Tensor(
            mul_tensor_1448, unsqueeze_default_827
        )
        mul_tensor_1448 = unsqueeze_default_827 = None
        relu_default_206 = torch.ops.aten.relu.default(add_tensor_913)
        add_tensor_913 = None
        convolution_default_382 = torch.ops.aten.convolution.default(
            relu_default_206,
            primals_795,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_383 = torch.ops.aten.convolution.default(
            convolution_default_382,
            primals_796,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_414 = torch.ops.aten.var.correction(
            convolution_default_383, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_207 = torch.ops.aten.mean.dim(convolution_default_383, [0, 2, 3], True)
        add_tensor_914 = torch.ops.aten.add.Tensor(var_correction_414, 0.001)
        var_correction_414 = None
        sqrt_default_207 = torch.ops.aten.sqrt.default(add_tensor_914)
        add_tensor_914 = None
        reciprocal_default_207 = torch.ops.aten.reciprocal.default(sqrt_default_207)
        sqrt_default_207 = None
        sub_tensor_207 = torch.ops.aten.sub.Tensor(
            convolution_default_383, mean_dim_207
        )
        mul_tensor_1449 = torch.ops.aten.mul.Tensor(
            sub_tensor_207, reciprocal_default_207
        )
        sub_tensor_207 = None
        squeeze_dim_1242 = torch.ops.aten.squeeze.dim(mean_dim_207, 3)
        mean_dim_207 = None
        squeeze_dim_1243 = torch.ops.aten.squeeze.dim(squeeze_dim_1242, 2)
        squeeze_dim_1242 = None
        squeeze_dim_1244 = torch.ops.aten.squeeze.dim(squeeze_dim_1243, 0)
        squeeze_dim_1243 = None
        squeeze_dim_1245 = torch.ops.aten.squeeze.dim(reciprocal_default_207, 3)
        reciprocal_default_207 = None
        squeeze_dim_1246 = torch.ops.aten.squeeze.dim(squeeze_dim_1245, 2)
        squeeze_dim_1245 = None
        squeeze_dim_1247 = torch.ops.aten.squeeze.dim(squeeze_dim_1246, 0)
        squeeze_dim_1246 = None
        unsqueeze_default_828 = torch.ops.aten.unsqueeze.default(primals_797, -1)
        unsqueeze_default_829 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_828, -1
        )
        unsqueeze_default_828 = None
        unsqueeze_default_830 = torch.ops.aten.unsqueeze.default(primals_798, -1)
        primals_798 = None
        unsqueeze_default_831 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_830, -1
        )
        unsqueeze_default_830 = None
        mul_tensor_1455 = torch.ops.aten.mul.Tensor(
            mul_tensor_1449, unsqueeze_default_829
        )
        mul_tensor_1449 = unsqueeze_default_829 = None
        add_tensor_917 = torch.ops.aten.add.Tensor(
            mul_tensor_1455, unsqueeze_default_831
        )
        mul_tensor_1455 = unsqueeze_default_831 = None
        relu_default_207 = torch.ops.aten.relu.default(add_tensor_905)
        convolution_default_384 = torch.ops.aten.convolution.default(
            relu_default_207,
            primals_799,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_385 = torch.ops.aten.convolution.default(
            convolution_default_384,
            primals_800,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_416 = torch.ops.aten.var.correction(
            convolution_default_385, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_208 = torch.ops.aten.mean.dim(convolution_default_385, [0, 2, 3], True)
        add_tensor_918 = torch.ops.aten.add.Tensor(var_correction_416, 0.001)
        var_correction_416 = None
        sqrt_default_208 = torch.ops.aten.sqrt.default(add_tensor_918)
        add_tensor_918 = None
        reciprocal_default_208 = torch.ops.aten.reciprocal.default(sqrt_default_208)
        sqrt_default_208 = None
        sub_tensor_208 = torch.ops.aten.sub.Tensor(
            convolution_default_385, mean_dim_208
        )
        mul_tensor_1456 = torch.ops.aten.mul.Tensor(
            sub_tensor_208, reciprocal_default_208
        )
        sub_tensor_208 = None
        squeeze_dim_1248 = torch.ops.aten.squeeze.dim(mean_dim_208, 3)
        mean_dim_208 = None
        squeeze_dim_1249 = torch.ops.aten.squeeze.dim(squeeze_dim_1248, 2)
        squeeze_dim_1248 = None
        squeeze_dim_1250 = torch.ops.aten.squeeze.dim(squeeze_dim_1249, 0)
        squeeze_dim_1249 = None
        squeeze_dim_1251 = torch.ops.aten.squeeze.dim(reciprocal_default_208, 3)
        reciprocal_default_208 = None
        squeeze_dim_1252 = torch.ops.aten.squeeze.dim(squeeze_dim_1251, 2)
        squeeze_dim_1251 = None
        squeeze_dim_1253 = torch.ops.aten.squeeze.dim(squeeze_dim_1252, 0)
        squeeze_dim_1252 = None
        unsqueeze_default_832 = torch.ops.aten.unsqueeze.default(primals_801, -1)
        unsqueeze_default_833 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_832, -1
        )
        unsqueeze_default_832 = None
        unsqueeze_default_834 = torch.ops.aten.unsqueeze.default(primals_802, -1)
        primals_802 = None
        unsqueeze_default_835 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_834, -1
        )
        unsqueeze_default_834 = None
        mul_tensor_1462 = torch.ops.aten.mul.Tensor(
            mul_tensor_1456, unsqueeze_default_833
        )
        mul_tensor_1456 = unsqueeze_default_833 = None
        add_tensor_921 = torch.ops.aten.add.Tensor(
            mul_tensor_1462, unsqueeze_default_835
        )
        mul_tensor_1462 = unsqueeze_default_835 = None
        relu_default_208 = torch.ops.aten.relu.default(add_tensor_921)
        add_tensor_921 = None
        convolution_default_386 = torch.ops.aten.convolution.default(
            relu_default_208,
            primals_803,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_387 = torch.ops.aten.convolution.default(
            convolution_default_386,
            primals_804,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_418 = torch.ops.aten.var.correction(
            convolution_default_387, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_209 = torch.ops.aten.mean.dim(convolution_default_387, [0, 2, 3], True)
        add_tensor_922 = torch.ops.aten.add.Tensor(var_correction_418, 0.001)
        var_correction_418 = None
        sqrt_default_209 = torch.ops.aten.sqrt.default(add_tensor_922)
        add_tensor_922 = None
        reciprocal_default_209 = torch.ops.aten.reciprocal.default(sqrt_default_209)
        sqrt_default_209 = None
        sub_tensor_209 = torch.ops.aten.sub.Tensor(
            convolution_default_387, mean_dim_209
        )
        mul_tensor_1463 = torch.ops.aten.mul.Tensor(
            sub_tensor_209, reciprocal_default_209
        )
        sub_tensor_209 = None
        squeeze_dim_1254 = torch.ops.aten.squeeze.dim(mean_dim_209, 3)
        mean_dim_209 = None
        squeeze_dim_1255 = torch.ops.aten.squeeze.dim(squeeze_dim_1254, 2)
        squeeze_dim_1254 = None
        squeeze_dim_1256 = torch.ops.aten.squeeze.dim(squeeze_dim_1255, 0)
        squeeze_dim_1255 = None
        squeeze_dim_1257 = torch.ops.aten.squeeze.dim(reciprocal_default_209, 3)
        reciprocal_default_209 = None
        squeeze_dim_1258 = torch.ops.aten.squeeze.dim(squeeze_dim_1257, 2)
        squeeze_dim_1257 = None
        squeeze_dim_1259 = torch.ops.aten.squeeze.dim(squeeze_dim_1258, 0)
        squeeze_dim_1258 = None
        unsqueeze_default_836 = torch.ops.aten.unsqueeze.default(primals_805, -1)
        unsqueeze_default_837 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_836, -1
        )
        unsqueeze_default_836 = None
        unsqueeze_default_838 = torch.ops.aten.unsqueeze.default(primals_806, -1)
        primals_806 = None
        unsqueeze_default_839 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_838, -1
        )
        unsqueeze_default_838 = None
        mul_tensor_1469 = torch.ops.aten.mul.Tensor(
            mul_tensor_1463, unsqueeze_default_837
        )
        mul_tensor_1463 = unsqueeze_default_837 = None
        add_tensor_925 = torch.ops.aten.add.Tensor(
            mul_tensor_1469, unsqueeze_default_839
        )
        mul_tensor_1469 = unsqueeze_default_839 = None
        add_tensor_926 = torch.ops.aten.add.Tensor(add_tensor_917, add_tensor_925)
        add_tensor_917 = add_tensor_925 = None
        convolution_default_388 = torch.ops.aten.convolution.default(
            relu_default_207,
            primals_807,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_389 = torch.ops.aten.convolution.default(
            convolution_default_388,
            primals_808,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_420 = torch.ops.aten.var.correction(
            convolution_default_389, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_210 = torch.ops.aten.mean.dim(convolution_default_389, [0, 2, 3], True)
        add_tensor_927 = torch.ops.aten.add.Tensor(var_correction_420, 0.001)
        var_correction_420 = None
        sqrt_default_210 = torch.ops.aten.sqrt.default(add_tensor_927)
        add_tensor_927 = None
        reciprocal_default_210 = torch.ops.aten.reciprocal.default(sqrt_default_210)
        sqrt_default_210 = None
        sub_tensor_210 = torch.ops.aten.sub.Tensor(
            convolution_default_389, mean_dim_210
        )
        mul_tensor_1470 = torch.ops.aten.mul.Tensor(
            sub_tensor_210, reciprocal_default_210
        )
        sub_tensor_210 = None
        squeeze_dim_1260 = torch.ops.aten.squeeze.dim(mean_dim_210, 3)
        mean_dim_210 = None
        squeeze_dim_1261 = torch.ops.aten.squeeze.dim(squeeze_dim_1260, 2)
        squeeze_dim_1260 = None
        squeeze_dim_1262 = torch.ops.aten.squeeze.dim(squeeze_dim_1261, 0)
        squeeze_dim_1261 = None
        squeeze_dim_1263 = torch.ops.aten.squeeze.dim(reciprocal_default_210, 3)
        reciprocal_default_210 = None
        squeeze_dim_1264 = torch.ops.aten.squeeze.dim(squeeze_dim_1263, 2)
        squeeze_dim_1263 = None
        squeeze_dim_1265 = torch.ops.aten.squeeze.dim(squeeze_dim_1264, 0)
        squeeze_dim_1264 = None
        unsqueeze_default_840 = torch.ops.aten.unsqueeze.default(primals_809, -1)
        unsqueeze_default_841 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_840, -1
        )
        unsqueeze_default_840 = None
        unsqueeze_default_842 = torch.ops.aten.unsqueeze.default(primals_810, -1)
        primals_810 = None
        unsqueeze_default_843 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_842, -1
        )
        unsqueeze_default_842 = None
        mul_tensor_1476 = torch.ops.aten.mul.Tensor(
            mul_tensor_1470, unsqueeze_default_841
        )
        mul_tensor_1470 = unsqueeze_default_841 = None
        add_tensor_930 = torch.ops.aten.add.Tensor(
            mul_tensor_1476, unsqueeze_default_843
        )
        mul_tensor_1476 = unsqueeze_default_843 = None
        relu_default_210 = torch.ops.aten.relu.default(add_tensor_930)
        add_tensor_930 = None
        convolution_default_390 = torch.ops.aten.convolution.default(
            relu_default_210,
            primals_811,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_391 = torch.ops.aten.convolution.default(
            convolution_default_390,
            primals_812,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_422 = torch.ops.aten.var.correction(
            convolution_default_391, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_211 = torch.ops.aten.mean.dim(convolution_default_391, [0, 2, 3], True)
        add_tensor_931 = torch.ops.aten.add.Tensor(var_correction_422, 0.001)
        var_correction_422 = None
        sqrt_default_211 = torch.ops.aten.sqrt.default(add_tensor_931)
        add_tensor_931 = None
        reciprocal_default_211 = torch.ops.aten.reciprocal.default(sqrt_default_211)
        sqrt_default_211 = None
        sub_tensor_211 = torch.ops.aten.sub.Tensor(
            convolution_default_391, mean_dim_211
        )
        mul_tensor_1477 = torch.ops.aten.mul.Tensor(
            sub_tensor_211, reciprocal_default_211
        )
        sub_tensor_211 = None
        squeeze_dim_1266 = torch.ops.aten.squeeze.dim(mean_dim_211, 3)
        mean_dim_211 = None
        squeeze_dim_1267 = torch.ops.aten.squeeze.dim(squeeze_dim_1266, 2)
        squeeze_dim_1266 = None
        squeeze_dim_1268 = torch.ops.aten.squeeze.dim(squeeze_dim_1267, 0)
        squeeze_dim_1267 = None
        squeeze_dim_1269 = torch.ops.aten.squeeze.dim(reciprocal_default_211, 3)
        reciprocal_default_211 = None
        squeeze_dim_1270 = torch.ops.aten.squeeze.dim(squeeze_dim_1269, 2)
        squeeze_dim_1269 = None
        squeeze_dim_1271 = torch.ops.aten.squeeze.dim(squeeze_dim_1270, 0)
        squeeze_dim_1270 = None
        unsqueeze_default_844 = torch.ops.aten.unsqueeze.default(primals_813, -1)
        unsqueeze_default_845 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_844, -1
        )
        unsqueeze_default_844 = None
        unsqueeze_default_846 = torch.ops.aten.unsqueeze.default(primals_814, -1)
        primals_814 = None
        unsqueeze_default_847 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_846, -1
        )
        unsqueeze_default_846 = None
        mul_tensor_1483 = torch.ops.aten.mul.Tensor(
            mul_tensor_1477, unsqueeze_default_845
        )
        mul_tensor_1477 = unsqueeze_default_845 = None
        add_tensor_934 = torch.ops.aten.add.Tensor(
            mul_tensor_1483, unsqueeze_default_847
        )
        mul_tensor_1483 = unsqueeze_default_847 = None
        convolution_default_392 = torch.ops.aten.convolution.default(
            relu_default_207,
            primals_815,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_207 = None
        convolution_default_393 = torch.ops.aten.convolution.default(
            convolution_default_392,
            primals_816,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_424 = torch.ops.aten.var.correction(
            convolution_default_393, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_212 = torch.ops.aten.mean.dim(convolution_default_393, [0, 2, 3], True)
        add_tensor_935 = torch.ops.aten.add.Tensor(var_correction_424, 0.001)
        var_correction_424 = None
        sqrt_default_212 = torch.ops.aten.sqrt.default(add_tensor_935)
        add_tensor_935 = None
        reciprocal_default_212 = torch.ops.aten.reciprocal.default(sqrt_default_212)
        sqrt_default_212 = None
        sub_tensor_212 = torch.ops.aten.sub.Tensor(
            convolution_default_393, mean_dim_212
        )
        mul_tensor_1484 = torch.ops.aten.mul.Tensor(
            sub_tensor_212, reciprocal_default_212
        )
        sub_tensor_212 = None
        squeeze_dim_1272 = torch.ops.aten.squeeze.dim(mean_dim_212, 3)
        mean_dim_212 = None
        squeeze_dim_1273 = torch.ops.aten.squeeze.dim(squeeze_dim_1272, 2)
        squeeze_dim_1272 = None
        squeeze_dim_1274 = torch.ops.aten.squeeze.dim(squeeze_dim_1273, 0)
        squeeze_dim_1273 = None
        squeeze_dim_1275 = torch.ops.aten.squeeze.dim(reciprocal_default_212, 3)
        reciprocal_default_212 = None
        squeeze_dim_1276 = torch.ops.aten.squeeze.dim(squeeze_dim_1275, 2)
        squeeze_dim_1275 = None
        squeeze_dim_1277 = torch.ops.aten.squeeze.dim(squeeze_dim_1276, 0)
        squeeze_dim_1276 = None
        unsqueeze_default_848 = torch.ops.aten.unsqueeze.default(primals_817, -1)
        unsqueeze_default_849 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_848, -1
        )
        unsqueeze_default_848 = None
        unsqueeze_default_850 = torch.ops.aten.unsqueeze.default(primals_818, -1)
        primals_818 = None
        unsqueeze_default_851 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_850, -1
        )
        unsqueeze_default_850 = None
        mul_tensor_1490 = torch.ops.aten.mul.Tensor(
            mul_tensor_1484, unsqueeze_default_849
        )
        mul_tensor_1484 = unsqueeze_default_849 = None
        add_tensor_938 = torch.ops.aten.add.Tensor(
            mul_tensor_1490, unsqueeze_default_851
        )
        mul_tensor_1490 = unsqueeze_default_851 = None
        relu_default_212 = torch.ops.aten.relu.default(add_tensor_938)
        add_tensor_938 = None
        convolution_default_394 = torch.ops.aten.convolution.default(
            relu_default_212,
            primals_819,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_395 = torch.ops.aten.convolution.default(
            convolution_default_394,
            primals_820,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_426 = torch.ops.aten.var.correction(
            convolution_default_395, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_213 = torch.ops.aten.mean.dim(convolution_default_395, [0, 2, 3], True)
        add_tensor_939 = torch.ops.aten.add.Tensor(var_correction_426, 0.001)
        var_correction_426 = None
        sqrt_default_213 = torch.ops.aten.sqrt.default(add_tensor_939)
        add_tensor_939 = None
        reciprocal_default_213 = torch.ops.aten.reciprocal.default(sqrt_default_213)
        sqrt_default_213 = None
        sub_tensor_213 = torch.ops.aten.sub.Tensor(
            convolution_default_395, mean_dim_213
        )
        mul_tensor_1491 = torch.ops.aten.mul.Tensor(
            sub_tensor_213, reciprocal_default_213
        )
        sub_tensor_213 = None
        squeeze_dim_1278 = torch.ops.aten.squeeze.dim(mean_dim_213, 3)
        mean_dim_213 = None
        squeeze_dim_1279 = torch.ops.aten.squeeze.dim(squeeze_dim_1278, 2)
        squeeze_dim_1278 = None
        squeeze_dim_1280 = torch.ops.aten.squeeze.dim(squeeze_dim_1279, 0)
        squeeze_dim_1279 = None
        squeeze_dim_1281 = torch.ops.aten.squeeze.dim(reciprocal_default_213, 3)
        reciprocal_default_213 = None
        squeeze_dim_1282 = torch.ops.aten.squeeze.dim(squeeze_dim_1281, 2)
        squeeze_dim_1281 = None
        squeeze_dim_1283 = torch.ops.aten.squeeze.dim(squeeze_dim_1282, 0)
        squeeze_dim_1282 = None
        unsqueeze_default_852 = torch.ops.aten.unsqueeze.default(primals_821, -1)
        unsqueeze_default_853 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_852, -1
        )
        unsqueeze_default_852 = None
        unsqueeze_default_854 = torch.ops.aten.unsqueeze.default(primals_822, -1)
        primals_822 = None
        unsqueeze_default_855 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_854, -1
        )
        unsqueeze_default_854 = None
        mul_tensor_1497 = torch.ops.aten.mul.Tensor(
            mul_tensor_1491, unsqueeze_default_853
        )
        mul_tensor_1491 = unsqueeze_default_853 = None
        add_tensor_942 = torch.ops.aten.add.Tensor(
            mul_tensor_1497, unsqueeze_default_855
        )
        mul_tensor_1497 = unsqueeze_default_855 = None
        add_tensor_943 = torch.ops.aten.add.Tensor(add_tensor_934, add_tensor_942)
        add_tensor_934 = add_tensor_942 = None
        avg_pool2d_default_55 = torch.ops.aten.avg_pool2d.default(
            add_tensor_909, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_944 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_55, add_tensor_905
        )
        avg_pool2d_default_55 = None
        avg_pool2d_default_56 = torch.ops.aten.avg_pool2d.default(
            add_tensor_905, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_945 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_56, avg_pool2d_default_56
        )
        avg_pool2d_default_56 = None
        convolution_default_396 = torch.ops.aten.convolution.default(
            relu_default_205,
            primals_823,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_205 = None
        convolution_default_397 = torch.ops.aten.convolution.default(
            convolution_default_396,
            primals_824,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_428 = torch.ops.aten.var.correction(
            convolution_default_397, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_214 = torch.ops.aten.mean.dim(convolution_default_397, [0, 2, 3], True)
        add_tensor_946 = torch.ops.aten.add.Tensor(var_correction_428, 0.001)
        var_correction_428 = None
        sqrt_default_214 = torch.ops.aten.sqrt.default(add_tensor_946)
        add_tensor_946 = None
        reciprocal_default_214 = torch.ops.aten.reciprocal.default(sqrt_default_214)
        sqrt_default_214 = None
        sub_tensor_214 = torch.ops.aten.sub.Tensor(
            convolution_default_397, mean_dim_214
        )
        mul_tensor_1498 = torch.ops.aten.mul.Tensor(
            sub_tensor_214, reciprocal_default_214
        )
        sub_tensor_214 = None
        squeeze_dim_1284 = torch.ops.aten.squeeze.dim(mean_dim_214, 3)
        mean_dim_214 = None
        squeeze_dim_1285 = torch.ops.aten.squeeze.dim(squeeze_dim_1284, 2)
        squeeze_dim_1284 = None
        squeeze_dim_1286 = torch.ops.aten.squeeze.dim(squeeze_dim_1285, 0)
        squeeze_dim_1285 = None
        squeeze_dim_1287 = torch.ops.aten.squeeze.dim(reciprocal_default_214, 3)
        reciprocal_default_214 = None
        squeeze_dim_1288 = torch.ops.aten.squeeze.dim(squeeze_dim_1287, 2)
        squeeze_dim_1287 = None
        squeeze_dim_1289 = torch.ops.aten.squeeze.dim(squeeze_dim_1288, 0)
        squeeze_dim_1288 = None
        unsqueeze_default_856 = torch.ops.aten.unsqueeze.default(primals_825, -1)
        unsqueeze_default_857 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_856, -1
        )
        unsqueeze_default_856 = None
        unsqueeze_default_858 = torch.ops.aten.unsqueeze.default(primals_826, -1)
        primals_826 = None
        unsqueeze_default_859 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_858, -1
        )
        unsqueeze_default_858 = None
        mul_tensor_1504 = torch.ops.aten.mul.Tensor(
            mul_tensor_1498, unsqueeze_default_857
        )
        mul_tensor_1498 = unsqueeze_default_857 = None
        add_tensor_949 = torch.ops.aten.add.Tensor(
            mul_tensor_1504, unsqueeze_default_859
        )
        mul_tensor_1504 = unsqueeze_default_859 = None
        relu_default_214 = torch.ops.aten.relu.default(add_tensor_949)
        add_tensor_949 = None
        convolution_default_398 = torch.ops.aten.convolution.default(
            relu_default_214,
            primals_827,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_399 = torch.ops.aten.convolution.default(
            convolution_default_398,
            primals_828,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_430 = torch.ops.aten.var.correction(
            convolution_default_399, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_215 = torch.ops.aten.mean.dim(convolution_default_399, [0, 2, 3], True)
        add_tensor_950 = torch.ops.aten.add.Tensor(var_correction_430, 0.001)
        var_correction_430 = None
        sqrt_default_215 = torch.ops.aten.sqrt.default(add_tensor_950)
        add_tensor_950 = None
        reciprocal_default_215 = torch.ops.aten.reciprocal.default(sqrt_default_215)
        sqrt_default_215 = None
        sub_tensor_215 = torch.ops.aten.sub.Tensor(
            convolution_default_399, mean_dim_215
        )
        mul_tensor_1505 = torch.ops.aten.mul.Tensor(
            sub_tensor_215, reciprocal_default_215
        )
        sub_tensor_215 = None
        squeeze_dim_1290 = torch.ops.aten.squeeze.dim(mean_dim_215, 3)
        mean_dim_215 = None
        squeeze_dim_1291 = torch.ops.aten.squeeze.dim(squeeze_dim_1290, 2)
        squeeze_dim_1290 = None
        squeeze_dim_1292 = torch.ops.aten.squeeze.dim(squeeze_dim_1291, 0)
        squeeze_dim_1291 = None
        squeeze_dim_1293 = torch.ops.aten.squeeze.dim(reciprocal_default_215, 3)
        reciprocal_default_215 = None
        squeeze_dim_1294 = torch.ops.aten.squeeze.dim(squeeze_dim_1293, 2)
        squeeze_dim_1293 = None
        squeeze_dim_1295 = torch.ops.aten.squeeze.dim(squeeze_dim_1294, 0)
        squeeze_dim_1294 = None
        unsqueeze_default_860 = torch.ops.aten.unsqueeze.default(primals_829, -1)
        unsqueeze_default_861 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_860, -1
        )
        unsqueeze_default_860 = None
        unsqueeze_default_862 = torch.ops.aten.unsqueeze.default(primals_830, -1)
        primals_830 = None
        unsqueeze_default_863 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_862, -1
        )
        unsqueeze_default_862 = None
        mul_tensor_1511 = torch.ops.aten.mul.Tensor(
            mul_tensor_1505, unsqueeze_default_861
        )
        mul_tensor_1505 = unsqueeze_default_861 = None
        add_tensor_953 = torch.ops.aten.add.Tensor(
            mul_tensor_1511, unsqueeze_default_863
        )
        mul_tensor_1511 = unsqueeze_default_863 = None
        add_tensor_954 = torch.ops.aten.add.Tensor(add_tensor_953, add_tensor_909)
        add_tensor_953 = add_tensor_909 = None
        cat_default_21 = torch.ops.aten.cat.default(
            [
                add_tensor_905,
                add_tensor_926,
                add_tensor_943,
                add_tensor_944,
                add_tensor_945,
                add_tensor_954,
            ],
            1,
        )
        add_tensor_905 = (
            add_tensor_926
        ) = add_tensor_943 = add_tensor_944 = add_tensor_945 = add_tensor_954 = None
        convolution_default_400 = torch.ops.aten.convolution.default(
            relu_default_204,
            primals_831,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_432 = torch.ops.aten.var.correction(
            convolution_default_400, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_216 = torch.ops.aten.mean.dim(convolution_default_400, [0, 2, 3], True)
        add_tensor_955 = torch.ops.aten.add.Tensor(var_correction_432, 0.001)
        var_correction_432 = None
        sqrt_default_216 = torch.ops.aten.sqrt.default(add_tensor_955)
        add_tensor_955 = None
        reciprocal_default_216 = torch.ops.aten.reciprocal.default(sqrt_default_216)
        sqrt_default_216 = None
        sub_tensor_216 = torch.ops.aten.sub.Tensor(
            convolution_default_400, mean_dim_216
        )
        mul_tensor_1512 = torch.ops.aten.mul.Tensor(
            sub_tensor_216, reciprocal_default_216
        )
        sub_tensor_216 = None
        unsqueeze_default_864 = torch.ops.aten.unsqueeze.default(primals_832, -1)
        unsqueeze_default_865 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_864, -1
        )
        unsqueeze_default_864 = None
        unsqueeze_default_866 = torch.ops.aten.unsqueeze.default(primals_833, -1)
        unsqueeze_default_867 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_866, -1
        )
        unsqueeze_default_866 = None
        mul_tensor_1518 = torch.ops.aten.mul.Tensor(
            mul_tensor_1512, unsqueeze_default_865
        )
        mul_tensor_1512 = unsqueeze_default_865 = None
        add_tensor_958 = torch.ops.aten.add.Tensor(
            mul_tensor_1518, unsqueeze_default_867
        )
        mul_tensor_1518 = unsqueeze_default_867 = None
        relu_default_216 = torch.ops.aten.relu.default(cat_default_21)
        cat_default_21 = None
        convolution_default_401 = torch.ops.aten.convolution.default(
            relu_default_216,
            primals_834,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_434 = torch.ops.aten.var.correction(
            convolution_default_401, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_217 = torch.ops.aten.mean.dim(convolution_default_401, [0, 2, 3], True)
        add_tensor_959 = torch.ops.aten.add.Tensor(var_correction_434, 0.001)
        var_correction_434 = None
        sqrt_default_217 = torch.ops.aten.sqrt.default(add_tensor_959)
        add_tensor_959 = None
        reciprocal_default_217 = torch.ops.aten.reciprocal.default(sqrt_default_217)
        sqrt_default_217 = None
        sub_tensor_217 = torch.ops.aten.sub.Tensor(
            convolution_default_401, mean_dim_217
        )
        mul_tensor_1519 = torch.ops.aten.mul.Tensor(
            sub_tensor_217, reciprocal_default_217
        )
        sub_tensor_217 = None
        unsqueeze_default_868 = torch.ops.aten.unsqueeze.default(primals_835, -1)
        unsqueeze_default_869 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_868, -1
        )
        unsqueeze_default_868 = None
        unsqueeze_default_870 = torch.ops.aten.unsqueeze.default(primals_836, -1)
        unsqueeze_default_871 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_870, -1
        )
        unsqueeze_default_870 = None
        mul_tensor_1525 = torch.ops.aten.mul.Tensor(
            mul_tensor_1519, unsqueeze_default_869
        )
        mul_tensor_1519 = unsqueeze_default_869 = None
        add_tensor_962 = torch.ops.aten.add.Tensor(
            mul_tensor_1525, unsqueeze_default_871
        )
        mul_tensor_1525 = unsqueeze_default_871 = None
        relu_default_217 = torch.ops.aten.relu.default(add_tensor_962)
        convolution_default_402 = torch.ops.aten.convolution.default(
            relu_default_217,
            primals_837,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_403 = torch.ops.aten.convolution.default(
            convolution_default_402,
            primals_838,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_436 = torch.ops.aten.var.correction(
            convolution_default_403, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_218 = torch.ops.aten.mean.dim(convolution_default_403, [0, 2, 3], True)
        add_tensor_963 = torch.ops.aten.add.Tensor(var_correction_436, 0.001)
        var_correction_436 = None
        sqrt_default_218 = torch.ops.aten.sqrt.default(add_tensor_963)
        add_tensor_963 = None
        reciprocal_default_218 = torch.ops.aten.reciprocal.default(sqrt_default_218)
        sqrt_default_218 = None
        sub_tensor_218 = torch.ops.aten.sub.Tensor(
            convolution_default_403, mean_dim_218
        )
        mul_tensor_1526 = torch.ops.aten.mul.Tensor(
            sub_tensor_218, reciprocal_default_218
        )
        sub_tensor_218 = None
        squeeze_dim_1308 = torch.ops.aten.squeeze.dim(mean_dim_218, 3)
        mean_dim_218 = None
        squeeze_dim_1309 = torch.ops.aten.squeeze.dim(squeeze_dim_1308, 2)
        squeeze_dim_1308 = None
        squeeze_dim_1310 = torch.ops.aten.squeeze.dim(squeeze_dim_1309, 0)
        squeeze_dim_1309 = None
        squeeze_dim_1311 = torch.ops.aten.squeeze.dim(reciprocal_default_218, 3)
        reciprocal_default_218 = None
        squeeze_dim_1312 = torch.ops.aten.squeeze.dim(squeeze_dim_1311, 2)
        squeeze_dim_1311 = None
        squeeze_dim_1313 = torch.ops.aten.squeeze.dim(squeeze_dim_1312, 0)
        squeeze_dim_1312 = None
        unsqueeze_default_872 = torch.ops.aten.unsqueeze.default(primals_839, -1)
        unsqueeze_default_873 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_872, -1
        )
        unsqueeze_default_872 = None
        unsqueeze_default_874 = torch.ops.aten.unsqueeze.default(primals_840, -1)
        primals_840 = None
        unsqueeze_default_875 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_874, -1
        )
        unsqueeze_default_874 = None
        mul_tensor_1532 = torch.ops.aten.mul.Tensor(
            mul_tensor_1526, unsqueeze_default_873
        )
        mul_tensor_1526 = unsqueeze_default_873 = None
        add_tensor_966 = torch.ops.aten.add.Tensor(
            mul_tensor_1532, unsqueeze_default_875
        )
        mul_tensor_1532 = unsqueeze_default_875 = None
        relu_default_218 = torch.ops.aten.relu.default(add_tensor_966)
        add_tensor_966 = None
        convolution_default_404 = torch.ops.aten.convolution.default(
            relu_default_218,
            primals_841,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_405 = torch.ops.aten.convolution.default(
            convolution_default_404,
            primals_842,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_438 = torch.ops.aten.var.correction(
            convolution_default_405, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_219 = torch.ops.aten.mean.dim(convolution_default_405, [0, 2, 3], True)
        add_tensor_967 = torch.ops.aten.add.Tensor(var_correction_438, 0.001)
        var_correction_438 = None
        sqrt_default_219 = torch.ops.aten.sqrt.default(add_tensor_967)
        add_tensor_967 = None
        reciprocal_default_219 = torch.ops.aten.reciprocal.default(sqrt_default_219)
        sqrt_default_219 = None
        sub_tensor_219 = torch.ops.aten.sub.Tensor(
            convolution_default_405, mean_dim_219
        )
        mul_tensor_1533 = torch.ops.aten.mul.Tensor(
            sub_tensor_219, reciprocal_default_219
        )
        sub_tensor_219 = None
        squeeze_dim_1314 = torch.ops.aten.squeeze.dim(mean_dim_219, 3)
        mean_dim_219 = None
        squeeze_dim_1315 = torch.ops.aten.squeeze.dim(squeeze_dim_1314, 2)
        squeeze_dim_1314 = None
        squeeze_dim_1316 = torch.ops.aten.squeeze.dim(squeeze_dim_1315, 0)
        squeeze_dim_1315 = None
        squeeze_dim_1317 = torch.ops.aten.squeeze.dim(reciprocal_default_219, 3)
        reciprocal_default_219 = None
        squeeze_dim_1318 = torch.ops.aten.squeeze.dim(squeeze_dim_1317, 2)
        squeeze_dim_1317 = None
        squeeze_dim_1319 = torch.ops.aten.squeeze.dim(squeeze_dim_1318, 0)
        squeeze_dim_1318 = None
        unsqueeze_default_876 = torch.ops.aten.unsqueeze.default(primals_843, -1)
        unsqueeze_default_877 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_876, -1
        )
        unsqueeze_default_876 = None
        unsqueeze_default_878 = torch.ops.aten.unsqueeze.default(primals_844, -1)
        primals_844 = None
        unsqueeze_default_879 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_878, -1
        )
        unsqueeze_default_878 = None
        mul_tensor_1539 = torch.ops.aten.mul.Tensor(
            mul_tensor_1533, unsqueeze_default_877
        )
        mul_tensor_1533 = unsqueeze_default_877 = None
        add_tensor_970 = torch.ops.aten.add.Tensor(
            mul_tensor_1539, unsqueeze_default_879
        )
        mul_tensor_1539 = unsqueeze_default_879 = None
        relu_default_219 = torch.ops.aten.relu.default(add_tensor_958)
        convolution_default_406 = torch.ops.aten.convolution.default(
            relu_default_219,
            primals_845,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_407 = torch.ops.aten.convolution.default(
            convolution_default_406,
            primals_846,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_440 = torch.ops.aten.var.correction(
            convolution_default_407, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_220 = torch.ops.aten.mean.dim(convolution_default_407, [0, 2, 3], True)
        add_tensor_971 = torch.ops.aten.add.Tensor(var_correction_440, 0.001)
        var_correction_440 = None
        sqrt_default_220 = torch.ops.aten.sqrt.default(add_tensor_971)
        add_tensor_971 = None
        reciprocal_default_220 = torch.ops.aten.reciprocal.default(sqrt_default_220)
        sqrt_default_220 = None
        sub_tensor_220 = torch.ops.aten.sub.Tensor(
            convolution_default_407, mean_dim_220
        )
        mul_tensor_1540 = torch.ops.aten.mul.Tensor(
            sub_tensor_220, reciprocal_default_220
        )
        sub_tensor_220 = None
        squeeze_dim_1320 = torch.ops.aten.squeeze.dim(mean_dim_220, 3)
        mean_dim_220 = None
        squeeze_dim_1321 = torch.ops.aten.squeeze.dim(squeeze_dim_1320, 2)
        squeeze_dim_1320 = None
        squeeze_dim_1322 = torch.ops.aten.squeeze.dim(squeeze_dim_1321, 0)
        squeeze_dim_1321 = None
        squeeze_dim_1323 = torch.ops.aten.squeeze.dim(reciprocal_default_220, 3)
        reciprocal_default_220 = None
        squeeze_dim_1324 = torch.ops.aten.squeeze.dim(squeeze_dim_1323, 2)
        squeeze_dim_1323 = None
        squeeze_dim_1325 = torch.ops.aten.squeeze.dim(squeeze_dim_1324, 0)
        squeeze_dim_1324 = None
        unsqueeze_default_880 = torch.ops.aten.unsqueeze.default(primals_847, -1)
        unsqueeze_default_881 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_880, -1
        )
        unsqueeze_default_880 = None
        unsqueeze_default_882 = torch.ops.aten.unsqueeze.default(primals_848, -1)
        primals_848 = None
        unsqueeze_default_883 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_882, -1
        )
        unsqueeze_default_882 = None
        mul_tensor_1546 = torch.ops.aten.mul.Tensor(
            mul_tensor_1540, unsqueeze_default_881
        )
        mul_tensor_1540 = unsqueeze_default_881 = None
        add_tensor_974 = torch.ops.aten.add.Tensor(
            mul_tensor_1546, unsqueeze_default_883
        )
        mul_tensor_1546 = unsqueeze_default_883 = None
        relu_default_220 = torch.ops.aten.relu.default(add_tensor_974)
        add_tensor_974 = None
        convolution_default_408 = torch.ops.aten.convolution.default(
            relu_default_220,
            primals_849,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_409 = torch.ops.aten.convolution.default(
            convolution_default_408,
            primals_850,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_442 = torch.ops.aten.var.correction(
            convolution_default_409, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_221 = torch.ops.aten.mean.dim(convolution_default_409, [0, 2, 3], True)
        add_tensor_975 = torch.ops.aten.add.Tensor(var_correction_442, 0.001)
        var_correction_442 = None
        sqrt_default_221 = torch.ops.aten.sqrt.default(add_tensor_975)
        add_tensor_975 = None
        reciprocal_default_221 = torch.ops.aten.reciprocal.default(sqrt_default_221)
        sqrt_default_221 = None
        sub_tensor_221 = torch.ops.aten.sub.Tensor(
            convolution_default_409, mean_dim_221
        )
        mul_tensor_1547 = torch.ops.aten.mul.Tensor(
            sub_tensor_221, reciprocal_default_221
        )
        sub_tensor_221 = None
        squeeze_dim_1326 = torch.ops.aten.squeeze.dim(mean_dim_221, 3)
        mean_dim_221 = None
        squeeze_dim_1327 = torch.ops.aten.squeeze.dim(squeeze_dim_1326, 2)
        squeeze_dim_1326 = None
        squeeze_dim_1328 = torch.ops.aten.squeeze.dim(squeeze_dim_1327, 0)
        squeeze_dim_1327 = None
        squeeze_dim_1329 = torch.ops.aten.squeeze.dim(reciprocal_default_221, 3)
        reciprocal_default_221 = None
        squeeze_dim_1330 = torch.ops.aten.squeeze.dim(squeeze_dim_1329, 2)
        squeeze_dim_1329 = None
        squeeze_dim_1331 = torch.ops.aten.squeeze.dim(squeeze_dim_1330, 0)
        squeeze_dim_1330 = None
        unsqueeze_default_884 = torch.ops.aten.unsqueeze.default(primals_851, -1)
        unsqueeze_default_885 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_884, -1
        )
        unsqueeze_default_884 = None
        unsqueeze_default_886 = torch.ops.aten.unsqueeze.default(primals_852, -1)
        primals_852 = None
        unsqueeze_default_887 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_886, -1
        )
        unsqueeze_default_886 = None
        mul_tensor_1553 = torch.ops.aten.mul.Tensor(
            mul_tensor_1547, unsqueeze_default_885
        )
        mul_tensor_1547 = unsqueeze_default_885 = None
        add_tensor_978 = torch.ops.aten.add.Tensor(
            mul_tensor_1553, unsqueeze_default_887
        )
        mul_tensor_1553 = unsqueeze_default_887 = None
        add_tensor_979 = torch.ops.aten.add.Tensor(add_tensor_970, add_tensor_978)
        add_tensor_970 = add_tensor_978 = None
        convolution_default_410 = torch.ops.aten.convolution.default(
            relu_default_219,
            primals_853,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_411 = torch.ops.aten.convolution.default(
            convolution_default_410,
            primals_854,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_444 = torch.ops.aten.var.correction(
            convolution_default_411, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_222 = torch.ops.aten.mean.dim(convolution_default_411, [0, 2, 3], True)
        add_tensor_980 = torch.ops.aten.add.Tensor(var_correction_444, 0.001)
        var_correction_444 = None
        sqrt_default_222 = torch.ops.aten.sqrt.default(add_tensor_980)
        add_tensor_980 = None
        reciprocal_default_222 = torch.ops.aten.reciprocal.default(sqrt_default_222)
        sqrt_default_222 = None
        sub_tensor_222 = torch.ops.aten.sub.Tensor(
            convolution_default_411, mean_dim_222
        )
        mul_tensor_1554 = torch.ops.aten.mul.Tensor(
            sub_tensor_222, reciprocal_default_222
        )
        sub_tensor_222 = None
        squeeze_dim_1332 = torch.ops.aten.squeeze.dim(mean_dim_222, 3)
        mean_dim_222 = None
        squeeze_dim_1333 = torch.ops.aten.squeeze.dim(squeeze_dim_1332, 2)
        squeeze_dim_1332 = None
        squeeze_dim_1334 = torch.ops.aten.squeeze.dim(squeeze_dim_1333, 0)
        squeeze_dim_1333 = None
        squeeze_dim_1335 = torch.ops.aten.squeeze.dim(reciprocal_default_222, 3)
        reciprocal_default_222 = None
        squeeze_dim_1336 = torch.ops.aten.squeeze.dim(squeeze_dim_1335, 2)
        squeeze_dim_1335 = None
        squeeze_dim_1337 = torch.ops.aten.squeeze.dim(squeeze_dim_1336, 0)
        squeeze_dim_1336 = None
        unsqueeze_default_888 = torch.ops.aten.unsqueeze.default(primals_855, -1)
        unsqueeze_default_889 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_888, -1
        )
        unsqueeze_default_888 = None
        unsqueeze_default_890 = torch.ops.aten.unsqueeze.default(primals_856, -1)
        primals_856 = None
        unsqueeze_default_891 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_890, -1
        )
        unsqueeze_default_890 = None
        mul_tensor_1560 = torch.ops.aten.mul.Tensor(
            mul_tensor_1554, unsqueeze_default_889
        )
        mul_tensor_1554 = unsqueeze_default_889 = None
        add_tensor_983 = torch.ops.aten.add.Tensor(
            mul_tensor_1560, unsqueeze_default_891
        )
        mul_tensor_1560 = unsqueeze_default_891 = None
        relu_default_222 = torch.ops.aten.relu.default(add_tensor_983)
        add_tensor_983 = None
        convolution_default_412 = torch.ops.aten.convolution.default(
            relu_default_222,
            primals_857,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_413 = torch.ops.aten.convolution.default(
            convolution_default_412,
            primals_858,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_446 = torch.ops.aten.var.correction(
            convolution_default_413, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_223 = torch.ops.aten.mean.dim(convolution_default_413, [0, 2, 3], True)
        add_tensor_984 = torch.ops.aten.add.Tensor(var_correction_446, 0.001)
        var_correction_446 = None
        sqrt_default_223 = torch.ops.aten.sqrt.default(add_tensor_984)
        add_tensor_984 = None
        reciprocal_default_223 = torch.ops.aten.reciprocal.default(sqrt_default_223)
        sqrt_default_223 = None
        sub_tensor_223 = torch.ops.aten.sub.Tensor(
            convolution_default_413, mean_dim_223
        )
        mul_tensor_1561 = torch.ops.aten.mul.Tensor(
            sub_tensor_223, reciprocal_default_223
        )
        sub_tensor_223 = None
        squeeze_dim_1338 = torch.ops.aten.squeeze.dim(mean_dim_223, 3)
        mean_dim_223 = None
        squeeze_dim_1339 = torch.ops.aten.squeeze.dim(squeeze_dim_1338, 2)
        squeeze_dim_1338 = None
        squeeze_dim_1340 = torch.ops.aten.squeeze.dim(squeeze_dim_1339, 0)
        squeeze_dim_1339 = None
        squeeze_dim_1341 = torch.ops.aten.squeeze.dim(reciprocal_default_223, 3)
        reciprocal_default_223 = None
        squeeze_dim_1342 = torch.ops.aten.squeeze.dim(squeeze_dim_1341, 2)
        squeeze_dim_1341 = None
        squeeze_dim_1343 = torch.ops.aten.squeeze.dim(squeeze_dim_1342, 0)
        squeeze_dim_1342 = None
        unsqueeze_default_892 = torch.ops.aten.unsqueeze.default(primals_859, -1)
        unsqueeze_default_893 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_892, -1
        )
        unsqueeze_default_892 = None
        unsqueeze_default_894 = torch.ops.aten.unsqueeze.default(primals_860, -1)
        primals_860 = None
        unsqueeze_default_895 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_894, -1
        )
        unsqueeze_default_894 = None
        mul_tensor_1567 = torch.ops.aten.mul.Tensor(
            mul_tensor_1561, unsqueeze_default_893
        )
        mul_tensor_1561 = unsqueeze_default_893 = None
        add_tensor_987 = torch.ops.aten.add.Tensor(
            mul_tensor_1567, unsqueeze_default_895
        )
        mul_tensor_1567 = unsqueeze_default_895 = None
        convolution_default_414 = torch.ops.aten.convolution.default(
            relu_default_219,
            primals_861,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_219 = None
        convolution_default_415 = torch.ops.aten.convolution.default(
            convolution_default_414,
            primals_862,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_448 = torch.ops.aten.var.correction(
            convolution_default_415, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_224 = torch.ops.aten.mean.dim(convolution_default_415, [0, 2, 3], True)
        add_tensor_988 = torch.ops.aten.add.Tensor(var_correction_448, 0.001)
        var_correction_448 = None
        sqrt_default_224 = torch.ops.aten.sqrt.default(add_tensor_988)
        add_tensor_988 = None
        reciprocal_default_224 = torch.ops.aten.reciprocal.default(sqrt_default_224)
        sqrt_default_224 = None
        sub_tensor_224 = torch.ops.aten.sub.Tensor(
            convolution_default_415, mean_dim_224
        )
        mul_tensor_1568 = torch.ops.aten.mul.Tensor(
            sub_tensor_224, reciprocal_default_224
        )
        sub_tensor_224 = None
        squeeze_dim_1344 = torch.ops.aten.squeeze.dim(mean_dim_224, 3)
        mean_dim_224 = None
        squeeze_dim_1345 = torch.ops.aten.squeeze.dim(squeeze_dim_1344, 2)
        squeeze_dim_1344 = None
        squeeze_dim_1346 = torch.ops.aten.squeeze.dim(squeeze_dim_1345, 0)
        squeeze_dim_1345 = None
        squeeze_dim_1347 = torch.ops.aten.squeeze.dim(reciprocal_default_224, 3)
        reciprocal_default_224 = None
        squeeze_dim_1348 = torch.ops.aten.squeeze.dim(squeeze_dim_1347, 2)
        squeeze_dim_1347 = None
        squeeze_dim_1349 = torch.ops.aten.squeeze.dim(squeeze_dim_1348, 0)
        squeeze_dim_1348 = None
        unsqueeze_default_896 = torch.ops.aten.unsqueeze.default(primals_863, -1)
        unsqueeze_default_897 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_896, -1
        )
        unsqueeze_default_896 = None
        unsqueeze_default_898 = torch.ops.aten.unsqueeze.default(primals_864, -1)
        primals_864 = None
        unsqueeze_default_899 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_898, -1
        )
        unsqueeze_default_898 = None
        mul_tensor_1574 = torch.ops.aten.mul.Tensor(
            mul_tensor_1568, unsqueeze_default_897
        )
        mul_tensor_1568 = unsqueeze_default_897 = None
        add_tensor_991 = torch.ops.aten.add.Tensor(
            mul_tensor_1574, unsqueeze_default_899
        )
        mul_tensor_1574 = unsqueeze_default_899 = None
        relu_default_224 = torch.ops.aten.relu.default(add_tensor_991)
        add_tensor_991 = None
        convolution_default_416 = torch.ops.aten.convolution.default(
            relu_default_224,
            primals_865,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_417 = torch.ops.aten.convolution.default(
            convolution_default_416,
            primals_866,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_450 = torch.ops.aten.var.correction(
            convolution_default_417, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_225 = torch.ops.aten.mean.dim(convolution_default_417, [0, 2, 3], True)
        add_tensor_992 = torch.ops.aten.add.Tensor(var_correction_450, 0.001)
        var_correction_450 = None
        sqrt_default_225 = torch.ops.aten.sqrt.default(add_tensor_992)
        add_tensor_992 = None
        reciprocal_default_225 = torch.ops.aten.reciprocal.default(sqrt_default_225)
        sqrt_default_225 = None
        sub_tensor_225 = torch.ops.aten.sub.Tensor(
            convolution_default_417, mean_dim_225
        )
        mul_tensor_1575 = torch.ops.aten.mul.Tensor(
            sub_tensor_225, reciprocal_default_225
        )
        sub_tensor_225 = None
        squeeze_dim_1350 = torch.ops.aten.squeeze.dim(mean_dim_225, 3)
        mean_dim_225 = None
        squeeze_dim_1351 = torch.ops.aten.squeeze.dim(squeeze_dim_1350, 2)
        squeeze_dim_1350 = None
        squeeze_dim_1352 = torch.ops.aten.squeeze.dim(squeeze_dim_1351, 0)
        squeeze_dim_1351 = None
        squeeze_dim_1353 = torch.ops.aten.squeeze.dim(reciprocal_default_225, 3)
        reciprocal_default_225 = None
        squeeze_dim_1354 = torch.ops.aten.squeeze.dim(squeeze_dim_1353, 2)
        squeeze_dim_1353 = None
        squeeze_dim_1355 = torch.ops.aten.squeeze.dim(squeeze_dim_1354, 0)
        squeeze_dim_1354 = None
        unsqueeze_default_900 = torch.ops.aten.unsqueeze.default(primals_867, -1)
        unsqueeze_default_901 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_900, -1
        )
        unsqueeze_default_900 = None
        unsqueeze_default_902 = torch.ops.aten.unsqueeze.default(primals_868, -1)
        primals_868 = None
        unsqueeze_default_903 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_902, -1
        )
        unsqueeze_default_902 = None
        mul_tensor_1581 = torch.ops.aten.mul.Tensor(
            mul_tensor_1575, unsqueeze_default_901
        )
        mul_tensor_1575 = unsqueeze_default_901 = None
        add_tensor_995 = torch.ops.aten.add.Tensor(
            mul_tensor_1581, unsqueeze_default_903
        )
        mul_tensor_1581 = unsqueeze_default_903 = None
        add_tensor_996 = torch.ops.aten.add.Tensor(add_tensor_987, add_tensor_995)
        add_tensor_987 = add_tensor_995 = None
        avg_pool2d_default_58 = torch.ops.aten.avg_pool2d.default(
            add_tensor_962, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_997 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_58, add_tensor_958
        )
        avg_pool2d_default_58 = None
        avg_pool2d_default_59 = torch.ops.aten.avg_pool2d.default(
            add_tensor_958, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_998 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_59, avg_pool2d_default_59
        )
        avg_pool2d_default_59 = None
        convolution_default_418 = torch.ops.aten.convolution.default(
            relu_default_217,
            primals_869,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_217 = None
        convolution_default_419 = torch.ops.aten.convolution.default(
            convolution_default_418,
            primals_870,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_452 = torch.ops.aten.var.correction(
            convolution_default_419, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_226 = torch.ops.aten.mean.dim(convolution_default_419, [0, 2, 3], True)
        add_tensor_999 = torch.ops.aten.add.Tensor(var_correction_452, 0.001)
        var_correction_452 = None
        sqrt_default_226 = torch.ops.aten.sqrt.default(add_tensor_999)
        add_tensor_999 = None
        reciprocal_default_226 = torch.ops.aten.reciprocal.default(sqrt_default_226)
        sqrt_default_226 = None
        sub_tensor_226 = torch.ops.aten.sub.Tensor(
            convolution_default_419, mean_dim_226
        )
        mul_tensor_1582 = torch.ops.aten.mul.Tensor(
            sub_tensor_226, reciprocal_default_226
        )
        sub_tensor_226 = None
        squeeze_dim_1356 = torch.ops.aten.squeeze.dim(mean_dim_226, 3)
        mean_dim_226 = None
        squeeze_dim_1357 = torch.ops.aten.squeeze.dim(squeeze_dim_1356, 2)
        squeeze_dim_1356 = None
        squeeze_dim_1358 = torch.ops.aten.squeeze.dim(squeeze_dim_1357, 0)
        squeeze_dim_1357 = None
        squeeze_dim_1359 = torch.ops.aten.squeeze.dim(reciprocal_default_226, 3)
        reciprocal_default_226 = None
        squeeze_dim_1360 = torch.ops.aten.squeeze.dim(squeeze_dim_1359, 2)
        squeeze_dim_1359 = None
        squeeze_dim_1361 = torch.ops.aten.squeeze.dim(squeeze_dim_1360, 0)
        squeeze_dim_1360 = None
        unsqueeze_default_904 = torch.ops.aten.unsqueeze.default(primals_871, -1)
        unsqueeze_default_905 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_904, -1
        )
        unsqueeze_default_904 = None
        unsqueeze_default_906 = torch.ops.aten.unsqueeze.default(primals_872, -1)
        primals_872 = None
        unsqueeze_default_907 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_906, -1
        )
        unsqueeze_default_906 = None
        mul_tensor_1588 = torch.ops.aten.mul.Tensor(
            mul_tensor_1582, unsqueeze_default_905
        )
        mul_tensor_1582 = unsqueeze_default_905 = None
        add_tensor_1002 = torch.ops.aten.add.Tensor(
            mul_tensor_1588, unsqueeze_default_907
        )
        mul_tensor_1588 = unsqueeze_default_907 = None
        relu_default_226 = torch.ops.aten.relu.default(add_tensor_1002)
        add_tensor_1002 = None
        convolution_default_420 = torch.ops.aten.convolution.default(
            relu_default_226,
            primals_873,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_421 = torch.ops.aten.convolution.default(
            convolution_default_420,
            primals_874,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_454 = torch.ops.aten.var.correction(
            convolution_default_421, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_227 = torch.ops.aten.mean.dim(convolution_default_421, [0, 2, 3], True)
        add_tensor_1003 = torch.ops.aten.add.Tensor(var_correction_454, 0.001)
        var_correction_454 = None
        sqrt_default_227 = torch.ops.aten.sqrt.default(add_tensor_1003)
        add_tensor_1003 = None
        reciprocal_default_227 = torch.ops.aten.reciprocal.default(sqrt_default_227)
        sqrt_default_227 = None
        sub_tensor_227 = torch.ops.aten.sub.Tensor(
            convolution_default_421, mean_dim_227
        )
        mul_tensor_1589 = torch.ops.aten.mul.Tensor(
            sub_tensor_227, reciprocal_default_227
        )
        sub_tensor_227 = None
        squeeze_dim_1362 = torch.ops.aten.squeeze.dim(mean_dim_227, 3)
        mean_dim_227 = None
        squeeze_dim_1363 = torch.ops.aten.squeeze.dim(squeeze_dim_1362, 2)
        squeeze_dim_1362 = None
        squeeze_dim_1364 = torch.ops.aten.squeeze.dim(squeeze_dim_1363, 0)
        squeeze_dim_1363 = None
        squeeze_dim_1365 = torch.ops.aten.squeeze.dim(reciprocal_default_227, 3)
        reciprocal_default_227 = None
        squeeze_dim_1366 = torch.ops.aten.squeeze.dim(squeeze_dim_1365, 2)
        squeeze_dim_1365 = None
        squeeze_dim_1367 = torch.ops.aten.squeeze.dim(squeeze_dim_1366, 0)
        squeeze_dim_1366 = None
        unsqueeze_default_908 = torch.ops.aten.unsqueeze.default(primals_875, -1)
        unsqueeze_default_909 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_908, -1
        )
        unsqueeze_default_908 = None
        unsqueeze_default_910 = torch.ops.aten.unsqueeze.default(primals_876, -1)
        primals_876 = None
        unsqueeze_default_911 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_910, -1
        )
        unsqueeze_default_910 = None
        mul_tensor_1595 = torch.ops.aten.mul.Tensor(
            mul_tensor_1589, unsqueeze_default_909
        )
        mul_tensor_1589 = unsqueeze_default_909 = None
        add_tensor_1006 = torch.ops.aten.add.Tensor(
            mul_tensor_1595, unsqueeze_default_911
        )
        mul_tensor_1595 = unsqueeze_default_911 = None
        add_tensor_1007 = torch.ops.aten.add.Tensor(add_tensor_1006, add_tensor_962)
        add_tensor_1006 = add_tensor_962 = None
        cat_default_22 = torch.ops.aten.cat.default(
            [
                add_tensor_958,
                add_tensor_979,
                add_tensor_996,
                add_tensor_997,
                add_tensor_998,
                add_tensor_1007,
            ],
            1,
        )
        add_tensor_958 = (
            add_tensor_979
        ) = add_tensor_996 = add_tensor_997 = add_tensor_998 = add_tensor_1007 = None
        convolution_default_422 = torch.ops.aten.convolution.default(
            relu_default_216,
            primals_877,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_456 = torch.ops.aten.var.correction(
            convolution_default_422, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_228 = torch.ops.aten.mean.dim(convolution_default_422, [0, 2, 3], True)
        add_tensor_1008 = torch.ops.aten.add.Tensor(var_correction_456, 0.001)
        var_correction_456 = None
        sqrt_default_228 = torch.ops.aten.sqrt.default(add_tensor_1008)
        add_tensor_1008 = None
        reciprocal_default_228 = torch.ops.aten.reciprocal.default(sqrt_default_228)
        sqrt_default_228 = None
        sub_tensor_228 = torch.ops.aten.sub.Tensor(
            convolution_default_422, mean_dim_228
        )
        mul_tensor_1596 = torch.ops.aten.mul.Tensor(
            sub_tensor_228, reciprocal_default_228
        )
        sub_tensor_228 = None
        unsqueeze_default_912 = torch.ops.aten.unsqueeze.default(primals_878, -1)
        unsqueeze_default_913 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_912, -1
        )
        unsqueeze_default_912 = None
        unsqueeze_default_914 = torch.ops.aten.unsqueeze.default(primals_879, -1)
        unsqueeze_default_915 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_914, -1
        )
        unsqueeze_default_914 = None
        mul_tensor_1602 = torch.ops.aten.mul.Tensor(
            mul_tensor_1596, unsqueeze_default_913
        )
        mul_tensor_1596 = unsqueeze_default_913 = None
        add_tensor_1011 = torch.ops.aten.add.Tensor(
            mul_tensor_1602, unsqueeze_default_915
        )
        mul_tensor_1602 = unsqueeze_default_915 = None
        relu_default_228 = torch.ops.aten.relu.default(cat_default_22)
        cat_default_22 = None
        convolution_default_423 = torch.ops.aten.convolution.default(
            relu_default_228,
            primals_880,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_458 = torch.ops.aten.var.correction(
            convolution_default_423, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_229 = torch.ops.aten.mean.dim(convolution_default_423, [0, 2, 3], True)
        add_tensor_1012 = torch.ops.aten.add.Tensor(var_correction_458, 0.001)
        var_correction_458 = None
        sqrt_default_229 = torch.ops.aten.sqrt.default(add_tensor_1012)
        add_tensor_1012 = None
        reciprocal_default_229 = torch.ops.aten.reciprocal.default(sqrt_default_229)
        sqrt_default_229 = None
        sub_tensor_229 = torch.ops.aten.sub.Tensor(
            convolution_default_423, mean_dim_229
        )
        mul_tensor_1603 = torch.ops.aten.mul.Tensor(
            sub_tensor_229, reciprocal_default_229
        )
        sub_tensor_229 = None
        unsqueeze_default_916 = torch.ops.aten.unsqueeze.default(primals_881, -1)
        unsqueeze_default_917 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_916, -1
        )
        unsqueeze_default_916 = None
        unsqueeze_default_918 = torch.ops.aten.unsqueeze.default(primals_882, -1)
        unsqueeze_default_919 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_918, -1
        )
        unsqueeze_default_918 = None
        mul_tensor_1609 = torch.ops.aten.mul.Tensor(
            mul_tensor_1603, unsqueeze_default_917
        )
        mul_tensor_1603 = unsqueeze_default_917 = None
        add_tensor_1015 = torch.ops.aten.add.Tensor(
            mul_tensor_1609, unsqueeze_default_919
        )
        mul_tensor_1609 = unsqueeze_default_919 = None
        relu_default_229 = torch.ops.aten.relu.default(add_tensor_1015)
        convolution_default_424 = torch.ops.aten.convolution.default(
            relu_default_229,
            primals_883,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_425 = torch.ops.aten.convolution.default(
            convolution_default_424,
            primals_884,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_460 = torch.ops.aten.var.correction(
            convolution_default_425, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_230 = torch.ops.aten.mean.dim(convolution_default_425, [0, 2, 3], True)
        add_tensor_1016 = torch.ops.aten.add.Tensor(var_correction_460, 0.001)
        var_correction_460 = None
        sqrt_default_230 = torch.ops.aten.sqrt.default(add_tensor_1016)
        add_tensor_1016 = None
        reciprocal_default_230 = torch.ops.aten.reciprocal.default(sqrt_default_230)
        sqrt_default_230 = None
        sub_tensor_230 = torch.ops.aten.sub.Tensor(
            convolution_default_425, mean_dim_230
        )
        mul_tensor_1610 = torch.ops.aten.mul.Tensor(
            sub_tensor_230, reciprocal_default_230
        )
        sub_tensor_230 = None
        squeeze_dim_1380 = torch.ops.aten.squeeze.dim(mean_dim_230, 3)
        mean_dim_230 = None
        squeeze_dim_1381 = torch.ops.aten.squeeze.dim(squeeze_dim_1380, 2)
        squeeze_dim_1380 = None
        squeeze_dim_1382 = torch.ops.aten.squeeze.dim(squeeze_dim_1381, 0)
        squeeze_dim_1381 = None
        squeeze_dim_1383 = torch.ops.aten.squeeze.dim(reciprocal_default_230, 3)
        reciprocal_default_230 = None
        squeeze_dim_1384 = torch.ops.aten.squeeze.dim(squeeze_dim_1383, 2)
        squeeze_dim_1383 = None
        squeeze_dim_1385 = torch.ops.aten.squeeze.dim(squeeze_dim_1384, 0)
        squeeze_dim_1384 = None
        unsqueeze_default_920 = torch.ops.aten.unsqueeze.default(primals_885, -1)
        unsqueeze_default_921 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_920, -1
        )
        unsqueeze_default_920 = None
        unsqueeze_default_922 = torch.ops.aten.unsqueeze.default(primals_886, -1)
        primals_886 = None
        unsqueeze_default_923 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_922, -1
        )
        unsqueeze_default_922 = None
        mul_tensor_1616 = torch.ops.aten.mul.Tensor(
            mul_tensor_1610, unsqueeze_default_921
        )
        mul_tensor_1610 = unsqueeze_default_921 = None
        add_tensor_1019 = torch.ops.aten.add.Tensor(
            mul_tensor_1616, unsqueeze_default_923
        )
        mul_tensor_1616 = unsqueeze_default_923 = None
        relu_default_230 = torch.ops.aten.relu.default(add_tensor_1019)
        add_tensor_1019 = None
        convolution_default_426 = torch.ops.aten.convolution.default(
            relu_default_230,
            primals_887,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_427 = torch.ops.aten.convolution.default(
            convolution_default_426,
            primals_888,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_462 = torch.ops.aten.var.correction(
            convolution_default_427, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_231 = torch.ops.aten.mean.dim(convolution_default_427, [0, 2, 3], True)
        add_tensor_1020 = torch.ops.aten.add.Tensor(var_correction_462, 0.001)
        var_correction_462 = None
        sqrt_default_231 = torch.ops.aten.sqrt.default(add_tensor_1020)
        add_tensor_1020 = None
        reciprocal_default_231 = torch.ops.aten.reciprocal.default(sqrt_default_231)
        sqrt_default_231 = None
        sub_tensor_231 = torch.ops.aten.sub.Tensor(
            convolution_default_427, mean_dim_231
        )
        mul_tensor_1617 = torch.ops.aten.mul.Tensor(
            sub_tensor_231, reciprocal_default_231
        )
        sub_tensor_231 = None
        squeeze_dim_1386 = torch.ops.aten.squeeze.dim(mean_dim_231, 3)
        mean_dim_231 = None
        squeeze_dim_1387 = torch.ops.aten.squeeze.dim(squeeze_dim_1386, 2)
        squeeze_dim_1386 = None
        squeeze_dim_1388 = torch.ops.aten.squeeze.dim(squeeze_dim_1387, 0)
        squeeze_dim_1387 = None
        squeeze_dim_1389 = torch.ops.aten.squeeze.dim(reciprocal_default_231, 3)
        reciprocal_default_231 = None
        squeeze_dim_1390 = torch.ops.aten.squeeze.dim(squeeze_dim_1389, 2)
        squeeze_dim_1389 = None
        squeeze_dim_1391 = torch.ops.aten.squeeze.dim(squeeze_dim_1390, 0)
        squeeze_dim_1390 = None
        unsqueeze_default_924 = torch.ops.aten.unsqueeze.default(primals_889, -1)
        unsqueeze_default_925 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_924, -1
        )
        unsqueeze_default_924 = None
        unsqueeze_default_926 = torch.ops.aten.unsqueeze.default(primals_890, -1)
        primals_890 = None
        unsqueeze_default_927 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_926, -1
        )
        unsqueeze_default_926 = None
        mul_tensor_1623 = torch.ops.aten.mul.Tensor(
            mul_tensor_1617, unsqueeze_default_925
        )
        mul_tensor_1617 = unsqueeze_default_925 = None
        add_tensor_1023 = torch.ops.aten.add.Tensor(
            mul_tensor_1623, unsqueeze_default_927
        )
        mul_tensor_1623 = unsqueeze_default_927 = None
        relu_default_231 = torch.ops.aten.relu.default(add_tensor_1011)
        convolution_default_428 = torch.ops.aten.convolution.default(
            relu_default_231,
            primals_891,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_429 = torch.ops.aten.convolution.default(
            convolution_default_428,
            primals_892,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_464 = torch.ops.aten.var.correction(
            convolution_default_429, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_232 = torch.ops.aten.mean.dim(convolution_default_429, [0, 2, 3], True)
        add_tensor_1024 = torch.ops.aten.add.Tensor(var_correction_464, 0.001)
        var_correction_464 = None
        sqrt_default_232 = torch.ops.aten.sqrt.default(add_tensor_1024)
        add_tensor_1024 = None
        reciprocal_default_232 = torch.ops.aten.reciprocal.default(sqrt_default_232)
        sqrt_default_232 = None
        sub_tensor_232 = torch.ops.aten.sub.Tensor(
            convolution_default_429, mean_dim_232
        )
        mul_tensor_1624 = torch.ops.aten.mul.Tensor(
            sub_tensor_232, reciprocal_default_232
        )
        sub_tensor_232 = None
        squeeze_dim_1392 = torch.ops.aten.squeeze.dim(mean_dim_232, 3)
        mean_dim_232 = None
        squeeze_dim_1393 = torch.ops.aten.squeeze.dim(squeeze_dim_1392, 2)
        squeeze_dim_1392 = None
        squeeze_dim_1394 = torch.ops.aten.squeeze.dim(squeeze_dim_1393, 0)
        squeeze_dim_1393 = None
        squeeze_dim_1395 = torch.ops.aten.squeeze.dim(reciprocal_default_232, 3)
        reciprocal_default_232 = None
        squeeze_dim_1396 = torch.ops.aten.squeeze.dim(squeeze_dim_1395, 2)
        squeeze_dim_1395 = None
        squeeze_dim_1397 = torch.ops.aten.squeeze.dim(squeeze_dim_1396, 0)
        squeeze_dim_1396 = None
        unsqueeze_default_928 = torch.ops.aten.unsqueeze.default(primals_893, -1)
        unsqueeze_default_929 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_928, -1
        )
        unsqueeze_default_928 = None
        unsqueeze_default_930 = torch.ops.aten.unsqueeze.default(primals_894, -1)
        primals_894 = None
        unsqueeze_default_931 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_930, -1
        )
        unsqueeze_default_930 = None
        mul_tensor_1630 = torch.ops.aten.mul.Tensor(
            mul_tensor_1624, unsqueeze_default_929
        )
        mul_tensor_1624 = unsqueeze_default_929 = None
        add_tensor_1027 = torch.ops.aten.add.Tensor(
            mul_tensor_1630, unsqueeze_default_931
        )
        mul_tensor_1630 = unsqueeze_default_931 = None
        relu_default_232 = torch.ops.aten.relu.default(add_tensor_1027)
        add_tensor_1027 = None
        convolution_default_430 = torch.ops.aten.convolution.default(
            relu_default_232,
            primals_895,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_431 = torch.ops.aten.convolution.default(
            convolution_default_430,
            primals_896,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_466 = torch.ops.aten.var.correction(
            convolution_default_431, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_233 = torch.ops.aten.mean.dim(convolution_default_431, [0, 2, 3], True)
        add_tensor_1028 = torch.ops.aten.add.Tensor(var_correction_466, 0.001)
        var_correction_466 = None
        sqrt_default_233 = torch.ops.aten.sqrt.default(add_tensor_1028)
        add_tensor_1028 = None
        reciprocal_default_233 = torch.ops.aten.reciprocal.default(sqrt_default_233)
        sqrt_default_233 = None
        sub_tensor_233 = torch.ops.aten.sub.Tensor(
            convolution_default_431, mean_dim_233
        )
        mul_tensor_1631 = torch.ops.aten.mul.Tensor(
            sub_tensor_233, reciprocal_default_233
        )
        sub_tensor_233 = None
        squeeze_dim_1398 = torch.ops.aten.squeeze.dim(mean_dim_233, 3)
        mean_dim_233 = None
        squeeze_dim_1399 = torch.ops.aten.squeeze.dim(squeeze_dim_1398, 2)
        squeeze_dim_1398 = None
        squeeze_dim_1400 = torch.ops.aten.squeeze.dim(squeeze_dim_1399, 0)
        squeeze_dim_1399 = None
        squeeze_dim_1401 = torch.ops.aten.squeeze.dim(reciprocal_default_233, 3)
        reciprocal_default_233 = None
        squeeze_dim_1402 = torch.ops.aten.squeeze.dim(squeeze_dim_1401, 2)
        squeeze_dim_1401 = None
        squeeze_dim_1403 = torch.ops.aten.squeeze.dim(squeeze_dim_1402, 0)
        squeeze_dim_1402 = None
        unsqueeze_default_932 = torch.ops.aten.unsqueeze.default(primals_897, -1)
        unsqueeze_default_933 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_932, -1
        )
        unsqueeze_default_932 = None
        unsqueeze_default_934 = torch.ops.aten.unsqueeze.default(primals_898, -1)
        primals_898 = None
        unsqueeze_default_935 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_934, -1
        )
        unsqueeze_default_934 = None
        mul_tensor_1637 = torch.ops.aten.mul.Tensor(
            mul_tensor_1631, unsqueeze_default_933
        )
        mul_tensor_1631 = unsqueeze_default_933 = None
        add_tensor_1031 = torch.ops.aten.add.Tensor(
            mul_tensor_1637, unsqueeze_default_935
        )
        mul_tensor_1637 = unsqueeze_default_935 = None
        add_tensor_1032 = torch.ops.aten.add.Tensor(add_tensor_1023, add_tensor_1031)
        add_tensor_1023 = add_tensor_1031 = None
        convolution_default_432 = torch.ops.aten.convolution.default(
            relu_default_231,
            primals_899,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_433 = torch.ops.aten.convolution.default(
            convolution_default_432,
            primals_900,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_468 = torch.ops.aten.var.correction(
            convolution_default_433, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_234 = torch.ops.aten.mean.dim(convolution_default_433, [0, 2, 3], True)
        add_tensor_1033 = torch.ops.aten.add.Tensor(var_correction_468, 0.001)
        var_correction_468 = None
        sqrt_default_234 = torch.ops.aten.sqrt.default(add_tensor_1033)
        add_tensor_1033 = None
        reciprocal_default_234 = torch.ops.aten.reciprocal.default(sqrt_default_234)
        sqrt_default_234 = None
        sub_tensor_234 = torch.ops.aten.sub.Tensor(
            convolution_default_433, mean_dim_234
        )
        mul_tensor_1638 = torch.ops.aten.mul.Tensor(
            sub_tensor_234, reciprocal_default_234
        )
        sub_tensor_234 = None
        squeeze_dim_1404 = torch.ops.aten.squeeze.dim(mean_dim_234, 3)
        mean_dim_234 = None
        squeeze_dim_1405 = torch.ops.aten.squeeze.dim(squeeze_dim_1404, 2)
        squeeze_dim_1404 = None
        squeeze_dim_1406 = torch.ops.aten.squeeze.dim(squeeze_dim_1405, 0)
        squeeze_dim_1405 = None
        squeeze_dim_1407 = torch.ops.aten.squeeze.dim(reciprocal_default_234, 3)
        reciprocal_default_234 = None
        squeeze_dim_1408 = torch.ops.aten.squeeze.dim(squeeze_dim_1407, 2)
        squeeze_dim_1407 = None
        squeeze_dim_1409 = torch.ops.aten.squeeze.dim(squeeze_dim_1408, 0)
        squeeze_dim_1408 = None
        unsqueeze_default_936 = torch.ops.aten.unsqueeze.default(primals_901, -1)
        unsqueeze_default_937 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_936, -1
        )
        unsqueeze_default_936 = None
        unsqueeze_default_938 = torch.ops.aten.unsqueeze.default(primals_902, -1)
        primals_902 = None
        unsqueeze_default_939 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_938, -1
        )
        unsqueeze_default_938 = None
        mul_tensor_1644 = torch.ops.aten.mul.Tensor(
            mul_tensor_1638, unsqueeze_default_937
        )
        mul_tensor_1638 = unsqueeze_default_937 = None
        add_tensor_1036 = torch.ops.aten.add.Tensor(
            mul_tensor_1644, unsqueeze_default_939
        )
        mul_tensor_1644 = unsqueeze_default_939 = None
        relu_default_234 = torch.ops.aten.relu.default(add_tensor_1036)
        add_tensor_1036 = None
        convolution_default_434 = torch.ops.aten.convolution.default(
            relu_default_234,
            primals_903,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_435 = torch.ops.aten.convolution.default(
            convolution_default_434,
            primals_904,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_470 = torch.ops.aten.var.correction(
            convolution_default_435, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_235 = torch.ops.aten.mean.dim(convolution_default_435, [0, 2, 3], True)
        add_tensor_1037 = torch.ops.aten.add.Tensor(var_correction_470, 0.001)
        var_correction_470 = None
        sqrt_default_235 = torch.ops.aten.sqrt.default(add_tensor_1037)
        add_tensor_1037 = None
        reciprocal_default_235 = torch.ops.aten.reciprocal.default(sqrt_default_235)
        sqrt_default_235 = None
        sub_tensor_235 = torch.ops.aten.sub.Tensor(
            convolution_default_435, mean_dim_235
        )
        mul_tensor_1645 = torch.ops.aten.mul.Tensor(
            sub_tensor_235, reciprocal_default_235
        )
        sub_tensor_235 = None
        squeeze_dim_1410 = torch.ops.aten.squeeze.dim(mean_dim_235, 3)
        mean_dim_235 = None
        squeeze_dim_1411 = torch.ops.aten.squeeze.dim(squeeze_dim_1410, 2)
        squeeze_dim_1410 = None
        squeeze_dim_1412 = torch.ops.aten.squeeze.dim(squeeze_dim_1411, 0)
        squeeze_dim_1411 = None
        squeeze_dim_1413 = torch.ops.aten.squeeze.dim(reciprocal_default_235, 3)
        reciprocal_default_235 = None
        squeeze_dim_1414 = torch.ops.aten.squeeze.dim(squeeze_dim_1413, 2)
        squeeze_dim_1413 = None
        squeeze_dim_1415 = torch.ops.aten.squeeze.dim(squeeze_dim_1414, 0)
        squeeze_dim_1414 = None
        unsqueeze_default_940 = torch.ops.aten.unsqueeze.default(primals_905, -1)
        unsqueeze_default_941 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_940, -1
        )
        unsqueeze_default_940 = None
        unsqueeze_default_942 = torch.ops.aten.unsqueeze.default(primals_906, -1)
        primals_906 = None
        unsqueeze_default_943 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_942, -1
        )
        unsqueeze_default_942 = None
        mul_tensor_1651 = torch.ops.aten.mul.Tensor(
            mul_tensor_1645, unsqueeze_default_941
        )
        mul_tensor_1645 = unsqueeze_default_941 = None
        add_tensor_1040 = torch.ops.aten.add.Tensor(
            mul_tensor_1651, unsqueeze_default_943
        )
        mul_tensor_1651 = unsqueeze_default_943 = None
        convolution_default_436 = torch.ops.aten.convolution.default(
            relu_default_231,
            primals_907,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_231 = None
        convolution_default_437 = torch.ops.aten.convolution.default(
            convolution_default_436,
            primals_908,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_472 = torch.ops.aten.var.correction(
            convolution_default_437, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_236 = torch.ops.aten.mean.dim(convolution_default_437, [0, 2, 3], True)
        add_tensor_1041 = torch.ops.aten.add.Tensor(var_correction_472, 0.001)
        var_correction_472 = None
        sqrt_default_236 = torch.ops.aten.sqrt.default(add_tensor_1041)
        add_tensor_1041 = None
        reciprocal_default_236 = torch.ops.aten.reciprocal.default(sqrt_default_236)
        sqrt_default_236 = None
        sub_tensor_236 = torch.ops.aten.sub.Tensor(
            convolution_default_437, mean_dim_236
        )
        mul_tensor_1652 = torch.ops.aten.mul.Tensor(
            sub_tensor_236, reciprocal_default_236
        )
        sub_tensor_236 = None
        squeeze_dim_1416 = torch.ops.aten.squeeze.dim(mean_dim_236, 3)
        mean_dim_236 = None
        squeeze_dim_1417 = torch.ops.aten.squeeze.dim(squeeze_dim_1416, 2)
        squeeze_dim_1416 = None
        squeeze_dim_1418 = torch.ops.aten.squeeze.dim(squeeze_dim_1417, 0)
        squeeze_dim_1417 = None
        squeeze_dim_1419 = torch.ops.aten.squeeze.dim(reciprocal_default_236, 3)
        reciprocal_default_236 = None
        squeeze_dim_1420 = torch.ops.aten.squeeze.dim(squeeze_dim_1419, 2)
        squeeze_dim_1419 = None
        squeeze_dim_1421 = torch.ops.aten.squeeze.dim(squeeze_dim_1420, 0)
        squeeze_dim_1420 = None
        unsqueeze_default_944 = torch.ops.aten.unsqueeze.default(primals_909, -1)
        unsqueeze_default_945 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_944, -1
        )
        unsqueeze_default_944 = None
        unsqueeze_default_946 = torch.ops.aten.unsqueeze.default(primals_910, -1)
        primals_910 = None
        unsqueeze_default_947 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_946, -1
        )
        unsqueeze_default_946 = None
        mul_tensor_1658 = torch.ops.aten.mul.Tensor(
            mul_tensor_1652, unsqueeze_default_945
        )
        mul_tensor_1652 = unsqueeze_default_945 = None
        add_tensor_1044 = torch.ops.aten.add.Tensor(
            mul_tensor_1658, unsqueeze_default_947
        )
        mul_tensor_1658 = unsqueeze_default_947 = None
        relu_default_236 = torch.ops.aten.relu.default(add_tensor_1044)
        add_tensor_1044 = None
        convolution_default_438 = torch.ops.aten.convolution.default(
            relu_default_236,
            primals_911,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_439 = torch.ops.aten.convolution.default(
            convolution_default_438,
            primals_912,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_474 = torch.ops.aten.var.correction(
            convolution_default_439, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_237 = torch.ops.aten.mean.dim(convolution_default_439, [0, 2, 3], True)
        add_tensor_1045 = torch.ops.aten.add.Tensor(var_correction_474, 0.001)
        var_correction_474 = None
        sqrt_default_237 = torch.ops.aten.sqrt.default(add_tensor_1045)
        add_tensor_1045 = None
        reciprocal_default_237 = torch.ops.aten.reciprocal.default(sqrt_default_237)
        sqrt_default_237 = None
        sub_tensor_237 = torch.ops.aten.sub.Tensor(
            convolution_default_439, mean_dim_237
        )
        mul_tensor_1659 = torch.ops.aten.mul.Tensor(
            sub_tensor_237, reciprocal_default_237
        )
        sub_tensor_237 = None
        squeeze_dim_1422 = torch.ops.aten.squeeze.dim(mean_dim_237, 3)
        mean_dim_237 = None
        squeeze_dim_1423 = torch.ops.aten.squeeze.dim(squeeze_dim_1422, 2)
        squeeze_dim_1422 = None
        squeeze_dim_1424 = torch.ops.aten.squeeze.dim(squeeze_dim_1423, 0)
        squeeze_dim_1423 = None
        squeeze_dim_1425 = torch.ops.aten.squeeze.dim(reciprocal_default_237, 3)
        reciprocal_default_237 = None
        squeeze_dim_1426 = torch.ops.aten.squeeze.dim(squeeze_dim_1425, 2)
        squeeze_dim_1425 = None
        squeeze_dim_1427 = torch.ops.aten.squeeze.dim(squeeze_dim_1426, 0)
        squeeze_dim_1426 = None
        unsqueeze_default_948 = torch.ops.aten.unsqueeze.default(primals_913, -1)
        unsqueeze_default_949 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_948, -1
        )
        unsqueeze_default_948 = None
        unsqueeze_default_950 = torch.ops.aten.unsqueeze.default(primals_914, -1)
        primals_914 = None
        unsqueeze_default_951 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_950, -1
        )
        unsqueeze_default_950 = None
        mul_tensor_1665 = torch.ops.aten.mul.Tensor(
            mul_tensor_1659, unsqueeze_default_949
        )
        mul_tensor_1659 = unsqueeze_default_949 = None
        add_tensor_1048 = torch.ops.aten.add.Tensor(
            mul_tensor_1665, unsqueeze_default_951
        )
        mul_tensor_1665 = unsqueeze_default_951 = None
        add_tensor_1049 = torch.ops.aten.add.Tensor(add_tensor_1040, add_tensor_1048)
        add_tensor_1040 = add_tensor_1048 = None
        avg_pool2d_default_61 = torch.ops.aten.avg_pool2d.default(
            add_tensor_1015, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_1050 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_61, add_tensor_1011
        )
        avg_pool2d_default_61 = None
        avg_pool2d_default_62 = torch.ops.aten.avg_pool2d.default(
            add_tensor_1011, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_1051 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_62, avg_pool2d_default_62
        )
        avg_pool2d_default_62 = None
        convolution_default_440 = torch.ops.aten.convolution.default(
            relu_default_229,
            primals_915,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_229 = None
        convolution_default_441 = torch.ops.aten.convolution.default(
            convolution_default_440,
            primals_916,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_476 = torch.ops.aten.var.correction(
            convolution_default_441, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_238 = torch.ops.aten.mean.dim(convolution_default_441, [0, 2, 3], True)
        add_tensor_1052 = torch.ops.aten.add.Tensor(var_correction_476, 0.001)
        var_correction_476 = None
        sqrt_default_238 = torch.ops.aten.sqrt.default(add_tensor_1052)
        add_tensor_1052 = None
        reciprocal_default_238 = torch.ops.aten.reciprocal.default(sqrt_default_238)
        sqrt_default_238 = None
        sub_tensor_238 = torch.ops.aten.sub.Tensor(
            convolution_default_441, mean_dim_238
        )
        mul_tensor_1666 = torch.ops.aten.mul.Tensor(
            sub_tensor_238, reciprocal_default_238
        )
        sub_tensor_238 = None
        squeeze_dim_1428 = torch.ops.aten.squeeze.dim(mean_dim_238, 3)
        mean_dim_238 = None
        squeeze_dim_1429 = torch.ops.aten.squeeze.dim(squeeze_dim_1428, 2)
        squeeze_dim_1428 = None
        squeeze_dim_1430 = torch.ops.aten.squeeze.dim(squeeze_dim_1429, 0)
        squeeze_dim_1429 = None
        squeeze_dim_1431 = torch.ops.aten.squeeze.dim(reciprocal_default_238, 3)
        reciprocal_default_238 = None
        squeeze_dim_1432 = torch.ops.aten.squeeze.dim(squeeze_dim_1431, 2)
        squeeze_dim_1431 = None
        squeeze_dim_1433 = torch.ops.aten.squeeze.dim(squeeze_dim_1432, 0)
        squeeze_dim_1432 = None
        unsqueeze_default_952 = torch.ops.aten.unsqueeze.default(primals_917, -1)
        unsqueeze_default_953 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_952, -1
        )
        unsqueeze_default_952 = None
        unsqueeze_default_954 = torch.ops.aten.unsqueeze.default(primals_918, -1)
        primals_918 = None
        unsqueeze_default_955 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_954, -1
        )
        unsqueeze_default_954 = None
        mul_tensor_1672 = torch.ops.aten.mul.Tensor(
            mul_tensor_1666, unsqueeze_default_953
        )
        mul_tensor_1666 = unsqueeze_default_953 = None
        add_tensor_1055 = torch.ops.aten.add.Tensor(
            mul_tensor_1672, unsqueeze_default_955
        )
        mul_tensor_1672 = unsqueeze_default_955 = None
        relu_default_238 = torch.ops.aten.relu.default(add_tensor_1055)
        add_tensor_1055 = None
        convolution_default_442 = torch.ops.aten.convolution.default(
            relu_default_238,
            primals_919,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_443 = torch.ops.aten.convolution.default(
            convolution_default_442,
            primals_920,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_478 = torch.ops.aten.var.correction(
            convolution_default_443, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_239 = torch.ops.aten.mean.dim(convolution_default_443, [0, 2, 3], True)
        add_tensor_1056 = torch.ops.aten.add.Tensor(var_correction_478, 0.001)
        var_correction_478 = None
        sqrt_default_239 = torch.ops.aten.sqrt.default(add_tensor_1056)
        add_tensor_1056 = None
        reciprocal_default_239 = torch.ops.aten.reciprocal.default(sqrt_default_239)
        sqrt_default_239 = None
        sub_tensor_239 = torch.ops.aten.sub.Tensor(
            convolution_default_443, mean_dim_239
        )
        mul_tensor_1673 = torch.ops.aten.mul.Tensor(
            sub_tensor_239, reciprocal_default_239
        )
        sub_tensor_239 = None
        squeeze_dim_1434 = torch.ops.aten.squeeze.dim(mean_dim_239, 3)
        mean_dim_239 = None
        squeeze_dim_1435 = torch.ops.aten.squeeze.dim(squeeze_dim_1434, 2)
        squeeze_dim_1434 = None
        squeeze_dim_1436 = torch.ops.aten.squeeze.dim(squeeze_dim_1435, 0)
        squeeze_dim_1435 = None
        squeeze_dim_1437 = torch.ops.aten.squeeze.dim(reciprocal_default_239, 3)
        reciprocal_default_239 = None
        squeeze_dim_1438 = torch.ops.aten.squeeze.dim(squeeze_dim_1437, 2)
        squeeze_dim_1437 = None
        squeeze_dim_1439 = torch.ops.aten.squeeze.dim(squeeze_dim_1438, 0)
        squeeze_dim_1438 = None
        unsqueeze_default_956 = torch.ops.aten.unsqueeze.default(primals_921, -1)
        unsqueeze_default_957 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_956, -1
        )
        unsqueeze_default_956 = None
        unsqueeze_default_958 = torch.ops.aten.unsqueeze.default(primals_922, -1)
        primals_922 = None
        unsqueeze_default_959 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_958, -1
        )
        unsqueeze_default_958 = None
        mul_tensor_1679 = torch.ops.aten.mul.Tensor(
            mul_tensor_1673, unsqueeze_default_957
        )
        mul_tensor_1673 = unsqueeze_default_957 = None
        add_tensor_1059 = torch.ops.aten.add.Tensor(
            mul_tensor_1679, unsqueeze_default_959
        )
        mul_tensor_1679 = unsqueeze_default_959 = None
        add_tensor_1060 = torch.ops.aten.add.Tensor(add_tensor_1059, add_tensor_1015)
        add_tensor_1059 = add_tensor_1015 = None
        cat_default_23 = torch.ops.aten.cat.default(
            [
                add_tensor_1011,
                add_tensor_1032,
                add_tensor_1049,
                add_tensor_1050,
                add_tensor_1051,
                add_tensor_1060,
            ],
            1,
        )
        add_tensor_1011 = (
            add_tensor_1032
        ) = add_tensor_1049 = add_tensor_1050 = add_tensor_1051 = add_tensor_1060 = None
        convolution_default_444 = torch.ops.aten.convolution.default(
            relu_default_228,
            primals_923,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_480 = torch.ops.aten.var.correction(
            convolution_default_444, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_240 = torch.ops.aten.mean.dim(convolution_default_444, [0, 2, 3], True)
        add_tensor_1061 = torch.ops.aten.add.Tensor(var_correction_480, 0.001)
        var_correction_480 = None
        sqrt_default_240 = torch.ops.aten.sqrt.default(add_tensor_1061)
        add_tensor_1061 = None
        reciprocal_default_240 = torch.ops.aten.reciprocal.default(sqrt_default_240)
        sqrt_default_240 = None
        sub_tensor_240 = torch.ops.aten.sub.Tensor(
            convolution_default_444, mean_dim_240
        )
        mul_tensor_1680 = torch.ops.aten.mul.Tensor(
            sub_tensor_240, reciprocal_default_240
        )
        sub_tensor_240 = None
        unsqueeze_default_960 = torch.ops.aten.unsqueeze.default(primals_924, -1)
        unsqueeze_default_961 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_960, -1
        )
        unsqueeze_default_960 = None
        unsqueeze_default_962 = torch.ops.aten.unsqueeze.default(primals_925, -1)
        unsqueeze_default_963 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_962, -1
        )
        unsqueeze_default_962 = None
        mul_tensor_1686 = torch.ops.aten.mul.Tensor(
            mul_tensor_1680, unsqueeze_default_961
        )
        mul_tensor_1680 = unsqueeze_default_961 = None
        add_tensor_1064 = torch.ops.aten.add.Tensor(
            mul_tensor_1686, unsqueeze_default_963
        )
        mul_tensor_1686 = unsqueeze_default_963 = None
        relu_default_240 = torch.ops.aten.relu.default(cat_default_23)
        cat_default_23 = None
        convolution_default_445 = torch.ops.aten.convolution.default(
            relu_default_240,
            primals_926,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_482 = torch.ops.aten.var.correction(
            convolution_default_445, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_241 = torch.ops.aten.mean.dim(convolution_default_445, [0, 2, 3], True)
        add_tensor_1065 = torch.ops.aten.add.Tensor(var_correction_482, 0.001)
        var_correction_482 = None
        sqrt_default_241 = torch.ops.aten.sqrt.default(add_tensor_1065)
        add_tensor_1065 = None
        reciprocal_default_241 = torch.ops.aten.reciprocal.default(sqrt_default_241)
        sqrt_default_241 = None
        sub_tensor_241 = torch.ops.aten.sub.Tensor(
            convolution_default_445, mean_dim_241
        )
        mul_tensor_1687 = torch.ops.aten.mul.Tensor(
            sub_tensor_241, reciprocal_default_241
        )
        sub_tensor_241 = None
        unsqueeze_default_964 = torch.ops.aten.unsqueeze.default(primals_927, -1)
        unsqueeze_default_965 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_964, -1
        )
        unsqueeze_default_964 = None
        unsqueeze_default_966 = torch.ops.aten.unsqueeze.default(primals_928, -1)
        unsqueeze_default_967 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_966, -1
        )
        unsqueeze_default_966 = None
        mul_tensor_1693 = torch.ops.aten.mul.Tensor(
            mul_tensor_1687, unsqueeze_default_965
        )
        mul_tensor_1687 = unsqueeze_default_965 = None
        add_tensor_1068 = torch.ops.aten.add.Tensor(
            mul_tensor_1693, unsqueeze_default_967
        )
        mul_tensor_1693 = unsqueeze_default_967 = None
        relu_default_241 = torch.ops.aten.relu.default(add_tensor_1068)
        convolution_default_446 = torch.ops.aten.convolution.default(
            relu_default_241,
            primals_929,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_447 = torch.ops.aten.convolution.default(
            convolution_default_446,
            primals_930,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_484 = torch.ops.aten.var.correction(
            convolution_default_447, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_242 = torch.ops.aten.mean.dim(convolution_default_447, [0, 2, 3], True)
        add_tensor_1069 = torch.ops.aten.add.Tensor(var_correction_484, 0.001)
        var_correction_484 = None
        sqrt_default_242 = torch.ops.aten.sqrt.default(add_tensor_1069)
        add_tensor_1069 = None
        reciprocal_default_242 = torch.ops.aten.reciprocal.default(sqrt_default_242)
        sqrt_default_242 = None
        sub_tensor_242 = torch.ops.aten.sub.Tensor(
            convolution_default_447, mean_dim_242
        )
        mul_tensor_1694 = torch.ops.aten.mul.Tensor(
            sub_tensor_242, reciprocal_default_242
        )
        sub_tensor_242 = None
        squeeze_dim_1452 = torch.ops.aten.squeeze.dim(mean_dim_242, 3)
        mean_dim_242 = None
        squeeze_dim_1453 = torch.ops.aten.squeeze.dim(squeeze_dim_1452, 2)
        squeeze_dim_1452 = None
        squeeze_dim_1454 = torch.ops.aten.squeeze.dim(squeeze_dim_1453, 0)
        squeeze_dim_1453 = None
        squeeze_dim_1455 = torch.ops.aten.squeeze.dim(reciprocal_default_242, 3)
        reciprocal_default_242 = None
        squeeze_dim_1456 = torch.ops.aten.squeeze.dim(squeeze_dim_1455, 2)
        squeeze_dim_1455 = None
        squeeze_dim_1457 = torch.ops.aten.squeeze.dim(squeeze_dim_1456, 0)
        squeeze_dim_1456 = None
        unsqueeze_default_968 = torch.ops.aten.unsqueeze.default(primals_931, -1)
        unsqueeze_default_969 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_968, -1
        )
        unsqueeze_default_968 = None
        unsqueeze_default_970 = torch.ops.aten.unsqueeze.default(primals_932, -1)
        primals_932 = None
        unsqueeze_default_971 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_970, -1
        )
        unsqueeze_default_970 = None
        mul_tensor_1700 = torch.ops.aten.mul.Tensor(
            mul_tensor_1694, unsqueeze_default_969
        )
        mul_tensor_1694 = unsqueeze_default_969 = None
        add_tensor_1072 = torch.ops.aten.add.Tensor(
            mul_tensor_1700, unsqueeze_default_971
        )
        mul_tensor_1700 = unsqueeze_default_971 = None
        relu_default_242 = torch.ops.aten.relu.default(add_tensor_1072)
        add_tensor_1072 = None
        convolution_default_448 = torch.ops.aten.convolution.default(
            relu_default_242,
            primals_933,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_449 = torch.ops.aten.convolution.default(
            convolution_default_448,
            primals_934,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_486 = torch.ops.aten.var.correction(
            convolution_default_449, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_243 = torch.ops.aten.mean.dim(convolution_default_449, [0, 2, 3], True)
        add_tensor_1073 = torch.ops.aten.add.Tensor(var_correction_486, 0.001)
        var_correction_486 = None
        sqrt_default_243 = torch.ops.aten.sqrt.default(add_tensor_1073)
        add_tensor_1073 = None
        reciprocal_default_243 = torch.ops.aten.reciprocal.default(sqrt_default_243)
        sqrt_default_243 = None
        sub_tensor_243 = torch.ops.aten.sub.Tensor(
            convolution_default_449, mean_dim_243
        )
        mul_tensor_1701 = torch.ops.aten.mul.Tensor(
            sub_tensor_243, reciprocal_default_243
        )
        sub_tensor_243 = None
        squeeze_dim_1458 = torch.ops.aten.squeeze.dim(mean_dim_243, 3)
        mean_dim_243 = None
        squeeze_dim_1459 = torch.ops.aten.squeeze.dim(squeeze_dim_1458, 2)
        squeeze_dim_1458 = None
        squeeze_dim_1460 = torch.ops.aten.squeeze.dim(squeeze_dim_1459, 0)
        squeeze_dim_1459 = None
        squeeze_dim_1461 = torch.ops.aten.squeeze.dim(reciprocal_default_243, 3)
        reciprocal_default_243 = None
        squeeze_dim_1462 = torch.ops.aten.squeeze.dim(squeeze_dim_1461, 2)
        squeeze_dim_1461 = None
        squeeze_dim_1463 = torch.ops.aten.squeeze.dim(squeeze_dim_1462, 0)
        squeeze_dim_1462 = None
        unsqueeze_default_972 = torch.ops.aten.unsqueeze.default(primals_935, -1)
        unsqueeze_default_973 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_972, -1
        )
        unsqueeze_default_972 = None
        unsqueeze_default_974 = torch.ops.aten.unsqueeze.default(primals_936, -1)
        primals_936 = None
        unsqueeze_default_975 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_974, -1
        )
        unsqueeze_default_974 = None
        mul_tensor_1707 = torch.ops.aten.mul.Tensor(
            mul_tensor_1701, unsqueeze_default_973
        )
        mul_tensor_1701 = unsqueeze_default_973 = None
        add_tensor_1076 = torch.ops.aten.add.Tensor(
            mul_tensor_1707, unsqueeze_default_975
        )
        mul_tensor_1707 = unsqueeze_default_975 = None
        relu_default_243 = torch.ops.aten.relu.default(add_tensor_1064)
        convolution_default_450 = torch.ops.aten.convolution.default(
            relu_default_243,
            primals_937,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_451 = torch.ops.aten.convolution.default(
            convolution_default_450,
            primals_938,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_488 = torch.ops.aten.var.correction(
            convolution_default_451, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_244 = torch.ops.aten.mean.dim(convolution_default_451, [0, 2, 3], True)
        add_tensor_1077 = torch.ops.aten.add.Tensor(var_correction_488, 0.001)
        var_correction_488 = None
        sqrt_default_244 = torch.ops.aten.sqrt.default(add_tensor_1077)
        add_tensor_1077 = None
        reciprocal_default_244 = torch.ops.aten.reciprocal.default(sqrt_default_244)
        sqrt_default_244 = None
        sub_tensor_244 = torch.ops.aten.sub.Tensor(
            convolution_default_451, mean_dim_244
        )
        mul_tensor_1708 = torch.ops.aten.mul.Tensor(
            sub_tensor_244, reciprocal_default_244
        )
        sub_tensor_244 = None
        squeeze_dim_1464 = torch.ops.aten.squeeze.dim(mean_dim_244, 3)
        mean_dim_244 = None
        squeeze_dim_1465 = torch.ops.aten.squeeze.dim(squeeze_dim_1464, 2)
        squeeze_dim_1464 = None
        squeeze_dim_1466 = torch.ops.aten.squeeze.dim(squeeze_dim_1465, 0)
        squeeze_dim_1465 = None
        squeeze_dim_1467 = torch.ops.aten.squeeze.dim(reciprocal_default_244, 3)
        reciprocal_default_244 = None
        squeeze_dim_1468 = torch.ops.aten.squeeze.dim(squeeze_dim_1467, 2)
        squeeze_dim_1467 = None
        squeeze_dim_1469 = torch.ops.aten.squeeze.dim(squeeze_dim_1468, 0)
        squeeze_dim_1468 = None
        unsqueeze_default_976 = torch.ops.aten.unsqueeze.default(primals_939, -1)
        unsqueeze_default_977 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_976, -1
        )
        unsqueeze_default_976 = None
        unsqueeze_default_978 = torch.ops.aten.unsqueeze.default(primals_940, -1)
        primals_940 = None
        unsqueeze_default_979 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_978, -1
        )
        unsqueeze_default_978 = None
        mul_tensor_1714 = torch.ops.aten.mul.Tensor(
            mul_tensor_1708, unsqueeze_default_977
        )
        mul_tensor_1708 = unsqueeze_default_977 = None
        add_tensor_1080 = torch.ops.aten.add.Tensor(
            mul_tensor_1714, unsqueeze_default_979
        )
        mul_tensor_1714 = unsqueeze_default_979 = None
        relu_default_244 = torch.ops.aten.relu.default(add_tensor_1080)
        add_tensor_1080 = None
        convolution_default_452 = torch.ops.aten.convolution.default(
            relu_default_244,
            primals_941,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_453 = torch.ops.aten.convolution.default(
            convolution_default_452,
            primals_942,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_490 = torch.ops.aten.var.correction(
            convolution_default_453, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_245 = torch.ops.aten.mean.dim(convolution_default_453, [0, 2, 3], True)
        add_tensor_1081 = torch.ops.aten.add.Tensor(var_correction_490, 0.001)
        var_correction_490 = None
        sqrt_default_245 = torch.ops.aten.sqrt.default(add_tensor_1081)
        add_tensor_1081 = None
        reciprocal_default_245 = torch.ops.aten.reciprocal.default(sqrt_default_245)
        sqrt_default_245 = None
        sub_tensor_245 = torch.ops.aten.sub.Tensor(
            convolution_default_453, mean_dim_245
        )
        mul_tensor_1715 = torch.ops.aten.mul.Tensor(
            sub_tensor_245, reciprocal_default_245
        )
        sub_tensor_245 = None
        squeeze_dim_1470 = torch.ops.aten.squeeze.dim(mean_dim_245, 3)
        mean_dim_245 = None
        squeeze_dim_1471 = torch.ops.aten.squeeze.dim(squeeze_dim_1470, 2)
        squeeze_dim_1470 = None
        squeeze_dim_1472 = torch.ops.aten.squeeze.dim(squeeze_dim_1471, 0)
        squeeze_dim_1471 = None
        squeeze_dim_1473 = torch.ops.aten.squeeze.dim(reciprocal_default_245, 3)
        reciprocal_default_245 = None
        squeeze_dim_1474 = torch.ops.aten.squeeze.dim(squeeze_dim_1473, 2)
        squeeze_dim_1473 = None
        squeeze_dim_1475 = torch.ops.aten.squeeze.dim(squeeze_dim_1474, 0)
        squeeze_dim_1474 = None
        unsqueeze_default_980 = torch.ops.aten.unsqueeze.default(primals_943, -1)
        unsqueeze_default_981 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_980, -1
        )
        unsqueeze_default_980 = None
        unsqueeze_default_982 = torch.ops.aten.unsqueeze.default(primals_944, -1)
        primals_944 = None
        unsqueeze_default_983 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_982, -1
        )
        unsqueeze_default_982 = None
        mul_tensor_1721 = torch.ops.aten.mul.Tensor(
            mul_tensor_1715, unsqueeze_default_981
        )
        mul_tensor_1715 = unsqueeze_default_981 = None
        add_tensor_1084 = torch.ops.aten.add.Tensor(
            mul_tensor_1721, unsqueeze_default_983
        )
        mul_tensor_1721 = unsqueeze_default_983 = None
        add_tensor_1085 = torch.ops.aten.add.Tensor(add_tensor_1076, add_tensor_1084)
        add_tensor_1076 = add_tensor_1084 = None
        convolution_default_454 = torch.ops.aten.convolution.default(
            relu_default_243,
            primals_945,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_455 = torch.ops.aten.convolution.default(
            convolution_default_454,
            primals_946,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_492 = torch.ops.aten.var.correction(
            convolution_default_455, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_246 = torch.ops.aten.mean.dim(convolution_default_455, [0, 2, 3], True)
        add_tensor_1086 = torch.ops.aten.add.Tensor(var_correction_492, 0.001)
        var_correction_492 = None
        sqrt_default_246 = torch.ops.aten.sqrt.default(add_tensor_1086)
        add_tensor_1086 = None
        reciprocal_default_246 = torch.ops.aten.reciprocal.default(sqrt_default_246)
        sqrt_default_246 = None
        sub_tensor_246 = torch.ops.aten.sub.Tensor(
            convolution_default_455, mean_dim_246
        )
        mul_tensor_1722 = torch.ops.aten.mul.Tensor(
            sub_tensor_246, reciprocal_default_246
        )
        sub_tensor_246 = None
        squeeze_dim_1476 = torch.ops.aten.squeeze.dim(mean_dim_246, 3)
        mean_dim_246 = None
        squeeze_dim_1477 = torch.ops.aten.squeeze.dim(squeeze_dim_1476, 2)
        squeeze_dim_1476 = None
        squeeze_dim_1478 = torch.ops.aten.squeeze.dim(squeeze_dim_1477, 0)
        squeeze_dim_1477 = None
        squeeze_dim_1479 = torch.ops.aten.squeeze.dim(reciprocal_default_246, 3)
        reciprocal_default_246 = None
        squeeze_dim_1480 = torch.ops.aten.squeeze.dim(squeeze_dim_1479, 2)
        squeeze_dim_1479 = None
        squeeze_dim_1481 = torch.ops.aten.squeeze.dim(squeeze_dim_1480, 0)
        squeeze_dim_1480 = None
        unsqueeze_default_984 = torch.ops.aten.unsqueeze.default(primals_947, -1)
        unsqueeze_default_985 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_984, -1
        )
        unsqueeze_default_984 = None
        unsqueeze_default_986 = torch.ops.aten.unsqueeze.default(primals_948, -1)
        primals_948 = None
        unsqueeze_default_987 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_986, -1
        )
        unsqueeze_default_986 = None
        mul_tensor_1728 = torch.ops.aten.mul.Tensor(
            mul_tensor_1722, unsqueeze_default_985
        )
        mul_tensor_1722 = unsqueeze_default_985 = None
        add_tensor_1089 = torch.ops.aten.add.Tensor(
            mul_tensor_1728, unsqueeze_default_987
        )
        mul_tensor_1728 = unsqueeze_default_987 = None
        relu_default_246 = torch.ops.aten.relu.default(add_tensor_1089)
        add_tensor_1089 = None
        convolution_default_456 = torch.ops.aten.convolution.default(
            relu_default_246,
            primals_949,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_457 = torch.ops.aten.convolution.default(
            convolution_default_456,
            primals_950,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_494 = torch.ops.aten.var.correction(
            convolution_default_457, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_247 = torch.ops.aten.mean.dim(convolution_default_457, [0, 2, 3], True)
        add_tensor_1090 = torch.ops.aten.add.Tensor(var_correction_494, 0.001)
        var_correction_494 = None
        sqrt_default_247 = torch.ops.aten.sqrt.default(add_tensor_1090)
        add_tensor_1090 = None
        reciprocal_default_247 = torch.ops.aten.reciprocal.default(sqrt_default_247)
        sqrt_default_247 = None
        sub_tensor_247 = torch.ops.aten.sub.Tensor(
            convolution_default_457, mean_dim_247
        )
        mul_tensor_1729 = torch.ops.aten.mul.Tensor(
            sub_tensor_247, reciprocal_default_247
        )
        sub_tensor_247 = None
        squeeze_dim_1482 = torch.ops.aten.squeeze.dim(mean_dim_247, 3)
        mean_dim_247 = None
        squeeze_dim_1483 = torch.ops.aten.squeeze.dim(squeeze_dim_1482, 2)
        squeeze_dim_1482 = None
        squeeze_dim_1484 = torch.ops.aten.squeeze.dim(squeeze_dim_1483, 0)
        squeeze_dim_1483 = None
        squeeze_dim_1485 = torch.ops.aten.squeeze.dim(reciprocal_default_247, 3)
        reciprocal_default_247 = None
        squeeze_dim_1486 = torch.ops.aten.squeeze.dim(squeeze_dim_1485, 2)
        squeeze_dim_1485 = None
        squeeze_dim_1487 = torch.ops.aten.squeeze.dim(squeeze_dim_1486, 0)
        squeeze_dim_1486 = None
        unsqueeze_default_988 = torch.ops.aten.unsqueeze.default(primals_951, -1)
        unsqueeze_default_989 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_988, -1
        )
        unsqueeze_default_988 = None
        unsqueeze_default_990 = torch.ops.aten.unsqueeze.default(primals_952, -1)
        primals_952 = None
        unsqueeze_default_991 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_990, -1
        )
        unsqueeze_default_990 = None
        mul_tensor_1735 = torch.ops.aten.mul.Tensor(
            mul_tensor_1729, unsqueeze_default_989
        )
        mul_tensor_1729 = unsqueeze_default_989 = None
        add_tensor_1093 = torch.ops.aten.add.Tensor(
            mul_tensor_1735, unsqueeze_default_991
        )
        mul_tensor_1735 = unsqueeze_default_991 = None
        convolution_default_458 = torch.ops.aten.convolution.default(
            relu_default_243,
            primals_953,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_243 = None
        convolution_default_459 = torch.ops.aten.convolution.default(
            convolution_default_458,
            primals_954,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_496 = torch.ops.aten.var.correction(
            convolution_default_459, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_248 = torch.ops.aten.mean.dim(convolution_default_459, [0, 2, 3], True)
        add_tensor_1094 = torch.ops.aten.add.Tensor(var_correction_496, 0.001)
        var_correction_496 = None
        sqrt_default_248 = torch.ops.aten.sqrt.default(add_tensor_1094)
        add_tensor_1094 = None
        reciprocal_default_248 = torch.ops.aten.reciprocal.default(sqrt_default_248)
        sqrt_default_248 = None
        sub_tensor_248 = torch.ops.aten.sub.Tensor(
            convolution_default_459, mean_dim_248
        )
        mul_tensor_1736 = torch.ops.aten.mul.Tensor(
            sub_tensor_248, reciprocal_default_248
        )
        sub_tensor_248 = None
        squeeze_dim_1488 = torch.ops.aten.squeeze.dim(mean_dim_248, 3)
        mean_dim_248 = None
        squeeze_dim_1489 = torch.ops.aten.squeeze.dim(squeeze_dim_1488, 2)
        squeeze_dim_1488 = None
        squeeze_dim_1490 = torch.ops.aten.squeeze.dim(squeeze_dim_1489, 0)
        squeeze_dim_1489 = None
        squeeze_dim_1491 = torch.ops.aten.squeeze.dim(reciprocal_default_248, 3)
        reciprocal_default_248 = None
        squeeze_dim_1492 = torch.ops.aten.squeeze.dim(squeeze_dim_1491, 2)
        squeeze_dim_1491 = None
        squeeze_dim_1493 = torch.ops.aten.squeeze.dim(squeeze_dim_1492, 0)
        squeeze_dim_1492 = None
        unsqueeze_default_992 = torch.ops.aten.unsqueeze.default(primals_955, -1)
        unsqueeze_default_993 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_992, -1
        )
        unsqueeze_default_992 = None
        unsqueeze_default_994 = torch.ops.aten.unsqueeze.default(primals_956, -1)
        primals_956 = None
        unsqueeze_default_995 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_994, -1
        )
        unsqueeze_default_994 = None
        mul_tensor_1742 = torch.ops.aten.mul.Tensor(
            mul_tensor_1736, unsqueeze_default_993
        )
        mul_tensor_1736 = unsqueeze_default_993 = None
        add_tensor_1097 = torch.ops.aten.add.Tensor(
            mul_tensor_1742, unsqueeze_default_995
        )
        mul_tensor_1742 = unsqueeze_default_995 = None
        relu_default_248 = torch.ops.aten.relu.default(add_tensor_1097)
        add_tensor_1097 = None
        convolution_default_460 = torch.ops.aten.convolution.default(
            relu_default_248,
            primals_957,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_461 = torch.ops.aten.convolution.default(
            convolution_default_460,
            primals_958,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_498 = torch.ops.aten.var.correction(
            convolution_default_461, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_249 = torch.ops.aten.mean.dim(convolution_default_461, [0, 2, 3], True)
        add_tensor_1098 = torch.ops.aten.add.Tensor(var_correction_498, 0.001)
        var_correction_498 = None
        sqrt_default_249 = torch.ops.aten.sqrt.default(add_tensor_1098)
        add_tensor_1098 = None
        reciprocal_default_249 = torch.ops.aten.reciprocal.default(sqrt_default_249)
        sqrt_default_249 = None
        sub_tensor_249 = torch.ops.aten.sub.Tensor(
            convolution_default_461, mean_dim_249
        )
        mul_tensor_1743 = torch.ops.aten.mul.Tensor(
            sub_tensor_249, reciprocal_default_249
        )
        sub_tensor_249 = None
        squeeze_dim_1494 = torch.ops.aten.squeeze.dim(mean_dim_249, 3)
        mean_dim_249 = None
        squeeze_dim_1495 = torch.ops.aten.squeeze.dim(squeeze_dim_1494, 2)
        squeeze_dim_1494 = None
        squeeze_dim_1496 = torch.ops.aten.squeeze.dim(squeeze_dim_1495, 0)
        squeeze_dim_1495 = None
        squeeze_dim_1497 = torch.ops.aten.squeeze.dim(reciprocal_default_249, 3)
        reciprocal_default_249 = None
        squeeze_dim_1498 = torch.ops.aten.squeeze.dim(squeeze_dim_1497, 2)
        squeeze_dim_1497 = None
        squeeze_dim_1499 = torch.ops.aten.squeeze.dim(squeeze_dim_1498, 0)
        squeeze_dim_1498 = None
        unsqueeze_default_996 = torch.ops.aten.unsqueeze.default(primals_959, -1)
        unsqueeze_default_997 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_996, -1
        )
        unsqueeze_default_996 = None
        unsqueeze_default_998 = torch.ops.aten.unsqueeze.default(primals_960, -1)
        primals_960 = None
        unsqueeze_default_999 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_998, -1
        )
        unsqueeze_default_998 = None
        mul_tensor_1749 = torch.ops.aten.mul.Tensor(
            mul_tensor_1743, unsqueeze_default_997
        )
        mul_tensor_1743 = unsqueeze_default_997 = None
        add_tensor_1101 = torch.ops.aten.add.Tensor(
            mul_tensor_1749, unsqueeze_default_999
        )
        mul_tensor_1749 = unsqueeze_default_999 = None
        add_tensor_1102 = torch.ops.aten.add.Tensor(add_tensor_1093, add_tensor_1101)
        add_tensor_1093 = add_tensor_1101 = None
        avg_pool2d_default_64 = torch.ops.aten.avg_pool2d.default(
            add_tensor_1068, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_1103 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_64, add_tensor_1064
        )
        avg_pool2d_default_64 = None
        avg_pool2d_default_65 = torch.ops.aten.avg_pool2d.default(
            add_tensor_1064, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_1104 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_65, avg_pool2d_default_65
        )
        avg_pool2d_default_65 = None
        convolution_default_462 = torch.ops.aten.convolution.default(
            relu_default_241,
            primals_961,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_241 = None
        convolution_default_463 = torch.ops.aten.convolution.default(
            convolution_default_462,
            primals_962,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_500 = torch.ops.aten.var.correction(
            convolution_default_463, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_250 = torch.ops.aten.mean.dim(convolution_default_463, [0, 2, 3], True)
        add_tensor_1105 = torch.ops.aten.add.Tensor(var_correction_500, 0.001)
        var_correction_500 = None
        sqrt_default_250 = torch.ops.aten.sqrt.default(add_tensor_1105)
        add_tensor_1105 = None
        reciprocal_default_250 = torch.ops.aten.reciprocal.default(sqrt_default_250)
        sqrt_default_250 = None
        sub_tensor_250 = torch.ops.aten.sub.Tensor(
            convolution_default_463, mean_dim_250
        )
        mul_tensor_1750 = torch.ops.aten.mul.Tensor(
            sub_tensor_250, reciprocal_default_250
        )
        sub_tensor_250 = None
        squeeze_dim_1500 = torch.ops.aten.squeeze.dim(mean_dim_250, 3)
        mean_dim_250 = None
        squeeze_dim_1501 = torch.ops.aten.squeeze.dim(squeeze_dim_1500, 2)
        squeeze_dim_1500 = None
        squeeze_dim_1502 = torch.ops.aten.squeeze.dim(squeeze_dim_1501, 0)
        squeeze_dim_1501 = None
        squeeze_dim_1503 = torch.ops.aten.squeeze.dim(reciprocal_default_250, 3)
        reciprocal_default_250 = None
        squeeze_dim_1504 = torch.ops.aten.squeeze.dim(squeeze_dim_1503, 2)
        squeeze_dim_1503 = None
        squeeze_dim_1505 = torch.ops.aten.squeeze.dim(squeeze_dim_1504, 0)
        squeeze_dim_1504 = None
        unsqueeze_default_1000 = torch.ops.aten.unsqueeze.default(primals_963, -1)
        unsqueeze_default_1001 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1000, -1
        )
        unsqueeze_default_1000 = None
        unsqueeze_default_1002 = torch.ops.aten.unsqueeze.default(primals_964, -1)
        primals_964 = None
        unsqueeze_default_1003 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1002, -1
        )
        unsqueeze_default_1002 = None
        mul_tensor_1756 = torch.ops.aten.mul.Tensor(
            mul_tensor_1750, unsqueeze_default_1001
        )
        mul_tensor_1750 = unsqueeze_default_1001 = None
        add_tensor_1108 = torch.ops.aten.add.Tensor(
            mul_tensor_1756, unsqueeze_default_1003
        )
        mul_tensor_1756 = unsqueeze_default_1003 = None
        relu_default_250 = torch.ops.aten.relu.default(add_tensor_1108)
        add_tensor_1108 = None
        convolution_default_464 = torch.ops.aten.convolution.default(
            relu_default_250,
            primals_965,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_465 = torch.ops.aten.convolution.default(
            convolution_default_464,
            primals_966,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_502 = torch.ops.aten.var.correction(
            convolution_default_465, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_251 = torch.ops.aten.mean.dim(convolution_default_465, [0, 2, 3], True)
        add_tensor_1109 = torch.ops.aten.add.Tensor(var_correction_502, 0.001)
        var_correction_502 = None
        sqrt_default_251 = torch.ops.aten.sqrt.default(add_tensor_1109)
        add_tensor_1109 = None
        reciprocal_default_251 = torch.ops.aten.reciprocal.default(sqrt_default_251)
        sqrt_default_251 = None
        sub_tensor_251 = torch.ops.aten.sub.Tensor(
            convolution_default_465, mean_dim_251
        )
        mul_tensor_1757 = torch.ops.aten.mul.Tensor(
            sub_tensor_251, reciprocal_default_251
        )
        sub_tensor_251 = None
        squeeze_dim_1506 = torch.ops.aten.squeeze.dim(mean_dim_251, 3)
        mean_dim_251 = None
        squeeze_dim_1507 = torch.ops.aten.squeeze.dim(squeeze_dim_1506, 2)
        squeeze_dim_1506 = None
        squeeze_dim_1508 = torch.ops.aten.squeeze.dim(squeeze_dim_1507, 0)
        squeeze_dim_1507 = None
        squeeze_dim_1509 = torch.ops.aten.squeeze.dim(reciprocal_default_251, 3)
        reciprocal_default_251 = None
        squeeze_dim_1510 = torch.ops.aten.squeeze.dim(squeeze_dim_1509, 2)
        squeeze_dim_1509 = None
        squeeze_dim_1511 = torch.ops.aten.squeeze.dim(squeeze_dim_1510, 0)
        squeeze_dim_1510 = None
        unsqueeze_default_1004 = torch.ops.aten.unsqueeze.default(primals_967, -1)
        unsqueeze_default_1005 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1004, -1
        )
        unsqueeze_default_1004 = None
        unsqueeze_default_1006 = torch.ops.aten.unsqueeze.default(primals_968, -1)
        primals_968 = None
        unsqueeze_default_1007 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1006, -1
        )
        unsqueeze_default_1006 = None
        mul_tensor_1763 = torch.ops.aten.mul.Tensor(
            mul_tensor_1757, unsqueeze_default_1005
        )
        mul_tensor_1757 = unsqueeze_default_1005 = None
        add_tensor_1112 = torch.ops.aten.add.Tensor(
            mul_tensor_1763, unsqueeze_default_1007
        )
        mul_tensor_1763 = unsqueeze_default_1007 = None
        add_tensor_1113 = torch.ops.aten.add.Tensor(add_tensor_1112, add_tensor_1068)
        add_tensor_1112 = add_tensor_1068 = None
        cat_default_24 = torch.ops.aten.cat.default(
            [
                add_tensor_1064,
                add_tensor_1085,
                add_tensor_1102,
                add_tensor_1103,
                add_tensor_1104,
                add_tensor_1113,
            ],
            1,
        )
        add_tensor_1064 = (
            add_tensor_1085
        ) = add_tensor_1102 = add_tensor_1103 = add_tensor_1104 = add_tensor_1113 = None
        convolution_default_466 = torch.ops.aten.convolution.default(
            relu_default_240,
            primals_969,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_504 = torch.ops.aten.var.correction(
            convolution_default_466, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_252 = torch.ops.aten.mean.dim(convolution_default_466, [0, 2, 3], True)
        add_tensor_1114 = torch.ops.aten.add.Tensor(var_correction_504, 0.001)
        var_correction_504 = None
        sqrt_default_252 = torch.ops.aten.sqrt.default(add_tensor_1114)
        add_tensor_1114 = None
        reciprocal_default_252 = torch.ops.aten.reciprocal.default(sqrt_default_252)
        sqrt_default_252 = None
        sub_tensor_252 = torch.ops.aten.sub.Tensor(
            convolution_default_466, mean_dim_252
        )
        mul_tensor_1764 = torch.ops.aten.mul.Tensor(
            sub_tensor_252, reciprocal_default_252
        )
        sub_tensor_252 = None
        unsqueeze_default_1008 = torch.ops.aten.unsqueeze.default(primals_970, -1)
        unsqueeze_default_1009 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1008, -1
        )
        unsqueeze_default_1008 = None
        unsqueeze_default_1010 = torch.ops.aten.unsqueeze.default(primals_971, -1)
        unsqueeze_default_1011 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1010, -1
        )
        unsqueeze_default_1010 = None
        mul_tensor_1770 = torch.ops.aten.mul.Tensor(
            mul_tensor_1764, unsqueeze_default_1009
        )
        mul_tensor_1764 = unsqueeze_default_1009 = None
        add_tensor_1117 = torch.ops.aten.add.Tensor(
            mul_tensor_1770, unsqueeze_default_1011
        )
        mul_tensor_1770 = unsqueeze_default_1011 = None
        relu_default_252 = torch.ops.aten.relu.default(cat_default_24)
        cat_default_24 = None
        convolution_default_467 = torch.ops.aten.convolution.default(
            relu_default_252,
            primals_972,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_506 = torch.ops.aten.var.correction(
            convolution_default_467, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_253 = torch.ops.aten.mean.dim(convolution_default_467, [0, 2, 3], True)
        add_tensor_1118 = torch.ops.aten.add.Tensor(var_correction_506, 0.001)
        var_correction_506 = None
        sqrt_default_253 = torch.ops.aten.sqrt.default(add_tensor_1118)
        add_tensor_1118 = None
        reciprocal_default_253 = torch.ops.aten.reciprocal.default(sqrt_default_253)
        sqrt_default_253 = None
        sub_tensor_253 = torch.ops.aten.sub.Tensor(
            convolution_default_467, mean_dim_253
        )
        mul_tensor_1771 = torch.ops.aten.mul.Tensor(
            sub_tensor_253, reciprocal_default_253
        )
        sub_tensor_253 = None
        unsqueeze_default_1012 = torch.ops.aten.unsqueeze.default(primals_973, -1)
        unsqueeze_default_1013 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1012, -1
        )
        unsqueeze_default_1012 = None
        unsqueeze_default_1014 = torch.ops.aten.unsqueeze.default(primals_974, -1)
        unsqueeze_default_1015 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1014, -1
        )
        unsqueeze_default_1014 = None
        mul_tensor_1777 = torch.ops.aten.mul.Tensor(
            mul_tensor_1771, unsqueeze_default_1013
        )
        mul_tensor_1771 = unsqueeze_default_1013 = None
        add_tensor_1121 = torch.ops.aten.add.Tensor(
            mul_tensor_1777, unsqueeze_default_1015
        )
        mul_tensor_1777 = unsqueeze_default_1015 = None
        relu_default_253 = torch.ops.aten.relu.default(add_tensor_1121)
        convolution_default_468 = torch.ops.aten.convolution.default(
            relu_default_253,
            primals_975,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_469 = torch.ops.aten.convolution.default(
            convolution_default_468,
            primals_976,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_508 = torch.ops.aten.var.correction(
            convolution_default_469, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_254 = torch.ops.aten.mean.dim(convolution_default_469, [0, 2, 3], True)
        add_tensor_1122 = torch.ops.aten.add.Tensor(var_correction_508, 0.001)
        var_correction_508 = None
        sqrt_default_254 = torch.ops.aten.sqrt.default(add_tensor_1122)
        add_tensor_1122 = None
        reciprocal_default_254 = torch.ops.aten.reciprocal.default(sqrt_default_254)
        sqrt_default_254 = None
        sub_tensor_254 = torch.ops.aten.sub.Tensor(
            convolution_default_469, mean_dim_254
        )
        mul_tensor_1778 = torch.ops.aten.mul.Tensor(
            sub_tensor_254, reciprocal_default_254
        )
        sub_tensor_254 = None
        squeeze_dim_1524 = torch.ops.aten.squeeze.dim(mean_dim_254, 3)
        mean_dim_254 = None
        squeeze_dim_1525 = torch.ops.aten.squeeze.dim(squeeze_dim_1524, 2)
        squeeze_dim_1524 = None
        squeeze_dim_1526 = torch.ops.aten.squeeze.dim(squeeze_dim_1525, 0)
        squeeze_dim_1525 = None
        squeeze_dim_1527 = torch.ops.aten.squeeze.dim(reciprocal_default_254, 3)
        reciprocal_default_254 = None
        squeeze_dim_1528 = torch.ops.aten.squeeze.dim(squeeze_dim_1527, 2)
        squeeze_dim_1527 = None
        squeeze_dim_1529 = torch.ops.aten.squeeze.dim(squeeze_dim_1528, 0)
        squeeze_dim_1528 = None
        unsqueeze_default_1016 = torch.ops.aten.unsqueeze.default(primals_977, -1)
        unsqueeze_default_1017 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1016, -1
        )
        unsqueeze_default_1016 = None
        unsqueeze_default_1018 = torch.ops.aten.unsqueeze.default(primals_978, -1)
        primals_978 = None
        unsqueeze_default_1019 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1018, -1
        )
        unsqueeze_default_1018 = None
        mul_tensor_1784 = torch.ops.aten.mul.Tensor(
            mul_tensor_1778, unsqueeze_default_1017
        )
        mul_tensor_1778 = unsqueeze_default_1017 = None
        add_tensor_1125 = torch.ops.aten.add.Tensor(
            mul_tensor_1784, unsqueeze_default_1019
        )
        mul_tensor_1784 = unsqueeze_default_1019 = None
        relu_default_254 = torch.ops.aten.relu.default(add_tensor_1125)
        add_tensor_1125 = None
        convolution_default_470 = torch.ops.aten.convolution.default(
            relu_default_254,
            primals_979,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_471 = torch.ops.aten.convolution.default(
            convolution_default_470,
            primals_980,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_510 = torch.ops.aten.var.correction(
            convolution_default_471, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_255 = torch.ops.aten.mean.dim(convolution_default_471, [0, 2, 3], True)
        add_tensor_1126 = torch.ops.aten.add.Tensor(var_correction_510, 0.001)
        var_correction_510 = None
        sqrt_default_255 = torch.ops.aten.sqrt.default(add_tensor_1126)
        add_tensor_1126 = None
        reciprocal_default_255 = torch.ops.aten.reciprocal.default(sqrt_default_255)
        sqrt_default_255 = None
        sub_tensor_255 = torch.ops.aten.sub.Tensor(
            convolution_default_471, mean_dim_255
        )
        mul_tensor_1785 = torch.ops.aten.mul.Tensor(
            sub_tensor_255, reciprocal_default_255
        )
        sub_tensor_255 = None
        squeeze_dim_1530 = torch.ops.aten.squeeze.dim(mean_dim_255, 3)
        mean_dim_255 = None
        squeeze_dim_1531 = torch.ops.aten.squeeze.dim(squeeze_dim_1530, 2)
        squeeze_dim_1530 = None
        squeeze_dim_1532 = torch.ops.aten.squeeze.dim(squeeze_dim_1531, 0)
        squeeze_dim_1531 = None
        squeeze_dim_1533 = torch.ops.aten.squeeze.dim(reciprocal_default_255, 3)
        reciprocal_default_255 = None
        squeeze_dim_1534 = torch.ops.aten.squeeze.dim(squeeze_dim_1533, 2)
        squeeze_dim_1533 = None
        squeeze_dim_1535 = torch.ops.aten.squeeze.dim(squeeze_dim_1534, 0)
        squeeze_dim_1534 = None
        unsqueeze_default_1020 = torch.ops.aten.unsqueeze.default(primals_981, -1)
        unsqueeze_default_1021 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1020, -1
        )
        unsqueeze_default_1020 = None
        unsqueeze_default_1022 = torch.ops.aten.unsqueeze.default(primals_982, -1)
        primals_982 = None
        unsqueeze_default_1023 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1022, -1
        )
        unsqueeze_default_1022 = None
        mul_tensor_1791 = torch.ops.aten.mul.Tensor(
            mul_tensor_1785, unsqueeze_default_1021
        )
        mul_tensor_1785 = unsqueeze_default_1021 = None
        add_tensor_1129 = torch.ops.aten.add.Tensor(
            mul_tensor_1791, unsqueeze_default_1023
        )
        mul_tensor_1791 = unsqueeze_default_1023 = None
        relu_default_255 = torch.ops.aten.relu.default(add_tensor_1117)
        convolution_default_472 = torch.ops.aten.convolution.default(
            relu_default_255,
            primals_983,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_473 = torch.ops.aten.convolution.default(
            convolution_default_472,
            primals_984,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_512 = torch.ops.aten.var.correction(
            convolution_default_473, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_256 = torch.ops.aten.mean.dim(convolution_default_473, [0, 2, 3], True)
        add_tensor_1130 = torch.ops.aten.add.Tensor(var_correction_512, 0.001)
        var_correction_512 = None
        sqrt_default_256 = torch.ops.aten.sqrt.default(add_tensor_1130)
        add_tensor_1130 = None
        reciprocal_default_256 = torch.ops.aten.reciprocal.default(sqrt_default_256)
        sqrt_default_256 = None
        sub_tensor_256 = torch.ops.aten.sub.Tensor(
            convolution_default_473, mean_dim_256
        )
        mul_tensor_1792 = torch.ops.aten.mul.Tensor(
            sub_tensor_256, reciprocal_default_256
        )
        sub_tensor_256 = None
        squeeze_dim_1536 = torch.ops.aten.squeeze.dim(mean_dim_256, 3)
        mean_dim_256 = None
        squeeze_dim_1537 = torch.ops.aten.squeeze.dim(squeeze_dim_1536, 2)
        squeeze_dim_1536 = None
        squeeze_dim_1538 = torch.ops.aten.squeeze.dim(squeeze_dim_1537, 0)
        squeeze_dim_1537 = None
        squeeze_dim_1539 = torch.ops.aten.squeeze.dim(reciprocal_default_256, 3)
        reciprocal_default_256 = None
        squeeze_dim_1540 = torch.ops.aten.squeeze.dim(squeeze_dim_1539, 2)
        squeeze_dim_1539 = None
        squeeze_dim_1541 = torch.ops.aten.squeeze.dim(squeeze_dim_1540, 0)
        squeeze_dim_1540 = None
        unsqueeze_default_1024 = torch.ops.aten.unsqueeze.default(primals_985, -1)
        unsqueeze_default_1025 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1024, -1
        )
        unsqueeze_default_1024 = None
        unsqueeze_default_1026 = torch.ops.aten.unsqueeze.default(primals_986, -1)
        primals_986 = None
        unsqueeze_default_1027 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1026, -1
        )
        unsqueeze_default_1026 = None
        mul_tensor_1798 = torch.ops.aten.mul.Tensor(
            mul_tensor_1792, unsqueeze_default_1025
        )
        mul_tensor_1792 = unsqueeze_default_1025 = None
        add_tensor_1133 = torch.ops.aten.add.Tensor(
            mul_tensor_1798, unsqueeze_default_1027
        )
        mul_tensor_1798 = unsqueeze_default_1027 = None
        relu_default_256 = torch.ops.aten.relu.default(add_tensor_1133)
        add_tensor_1133 = None
        convolution_default_474 = torch.ops.aten.convolution.default(
            relu_default_256,
            primals_987,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_475 = torch.ops.aten.convolution.default(
            convolution_default_474,
            primals_988,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_514 = torch.ops.aten.var.correction(
            convolution_default_475, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_257 = torch.ops.aten.mean.dim(convolution_default_475, [0, 2, 3], True)
        add_tensor_1134 = torch.ops.aten.add.Tensor(var_correction_514, 0.001)
        var_correction_514 = None
        sqrt_default_257 = torch.ops.aten.sqrt.default(add_tensor_1134)
        add_tensor_1134 = None
        reciprocal_default_257 = torch.ops.aten.reciprocal.default(sqrt_default_257)
        sqrt_default_257 = None
        sub_tensor_257 = torch.ops.aten.sub.Tensor(
            convolution_default_475, mean_dim_257
        )
        mul_tensor_1799 = torch.ops.aten.mul.Tensor(
            sub_tensor_257, reciprocal_default_257
        )
        sub_tensor_257 = None
        squeeze_dim_1542 = torch.ops.aten.squeeze.dim(mean_dim_257, 3)
        mean_dim_257 = None
        squeeze_dim_1543 = torch.ops.aten.squeeze.dim(squeeze_dim_1542, 2)
        squeeze_dim_1542 = None
        squeeze_dim_1544 = torch.ops.aten.squeeze.dim(squeeze_dim_1543, 0)
        squeeze_dim_1543 = None
        squeeze_dim_1545 = torch.ops.aten.squeeze.dim(reciprocal_default_257, 3)
        reciprocal_default_257 = None
        squeeze_dim_1546 = torch.ops.aten.squeeze.dim(squeeze_dim_1545, 2)
        squeeze_dim_1545 = None
        squeeze_dim_1547 = torch.ops.aten.squeeze.dim(squeeze_dim_1546, 0)
        squeeze_dim_1546 = None
        unsqueeze_default_1028 = torch.ops.aten.unsqueeze.default(primals_989, -1)
        unsqueeze_default_1029 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1028, -1
        )
        unsqueeze_default_1028 = None
        unsqueeze_default_1030 = torch.ops.aten.unsqueeze.default(primals_990, -1)
        primals_990 = None
        unsqueeze_default_1031 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1030, -1
        )
        unsqueeze_default_1030 = None
        mul_tensor_1805 = torch.ops.aten.mul.Tensor(
            mul_tensor_1799, unsqueeze_default_1029
        )
        mul_tensor_1799 = unsqueeze_default_1029 = None
        add_tensor_1137 = torch.ops.aten.add.Tensor(
            mul_tensor_1805, unsqueeze_default_1031
        )
        mul_tensor_1805 = unsqueeze_default_1031 = None
        add_tensor_1138 = torch.ops.aten.add.Tensor(add_tensor_1129, add_tensor_1137)
        add_tensor_1129 = add_tensor_1137 = None
        convolution_default_476 = torch.ops.aten.convolution.default(
            relu_default_255,
            primals_991,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_477 = torch.ops.aten.convolution.default(
            convolution_default_476,
            primals_992,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_516 = torch.ops.aten.var.correction(
            convolution_default_477, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_258 = torch.ops.aten.mean.dim(convolution_default_477, [0, 2, 3], True)
        add_tensor_1139 = torch.ops.aten.add.Tensor(var_correction_516, 0.001)
        var_correction_516 = None
        sqrt_default_258 = torch.ops.aten.sqrt.default(add_tensor_1139)
        add_tensor_1139 = None
        reciprocal_default_258 = torch.ops.aten.reciprocal.default(sqrt_default_258)
        sqrt_default_258 = None
        sub_tensor_258 = torch.ops.aten.sub.Tensor(
            convolution_default_477, mean_dim_258
        )
        mul_tensor_1806 = torch.ops.aten.mul.Tensor(
            sub_tensor_258, reciprocal_default_258
        )
        sub_tensor_258 = None
        squeeze_dim_1548 = torch.ops.aten.squeeze.dim(mean_dim_258, 3)
        mean_dim_258 = None
        squeeze_dim_1549 = torch.ops.aten.squeeze.dim(squeeze_dim_1548, 2)
        squeeze_dim_1548 = None
        squeeze_dim_1550 = torch.ops.aten.squeeze.dim(squeeze_dim_1549, 0)
        squeeze_dim_1549 = None
        squeeze_dim_1551 = torch.ops.aten.squeeze.dim(reciprocal_default_258, 3)
        reciprocal_default_258 = None
        squeeze_dim_1552 = torch.ops.aten.squeeze.dim(squeeze_dim_1551, 2)
        squeeze_dim_1551 = None
        squeeze_dim_1553 = torch.ops.aten.squeeze.dim(squeeze_dim_1552, 0)
        squeeze_dim_1552 = None
        unsqueeze_default_1032 = torch.ops.aten.unsqueeze.default(primals_993, -1)
        unsqueeze_default_1033 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1032, -1
        )
        unsqueeze_default_1032 = None
        unsqueeze_default_1034 = torch.ops.aten.unsqueeze.default(primals_994, -1)
        primals_994 = None
        unsqueeze_default_1035 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1034, -1
        )
        unsqueeze_default_1034 = None
        mul_tensor_1812 = torch.ops.aten.mul.Tensor(
            mul_tensor_1806, unsqueeze_default_1033
        )
        mul_tensor_1806 = unsqueeze_default_1033 = None
        add_tensor_1142 = torch.ops.aten.add.Tensor(
            mul_tensor_1812, unsqueeze_default_1035
        )
        mul_tensor_1812 = unsqueeze_default_1035 = None
        relu_default_258 = torch.ops.aten.relu.default(add_tensor_1142)
        add_tensor_1142 = None
        convolution_default_478 = torch.ops.aten.convolution.default(
            relu_default_258,
            primals_995,
            None,
            [1, 1],
            [2, 2],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_479 = torch.ops.aten.convolution.default(
            convolution_default_478,
            primals_996,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_518 = torch.ops.aten.var.correction(
            convolution_default_479, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_259 = torch.ops.aten.mean.dim(convolution_default_479, [0, 2, 3], True)
        add_tensor_1143 = torch.ops.aten.add.Tensor(var_correction_518, 0.001)
        var_correction_518 = None
        sqrt_default_259 = torch.ops.aten.sqrt.default(add_tensor_1143)
        add_tensor_1143 = None
        reciprocal_default_259 = torch.ops.aten.reciprocal.default(sqrt_default_259)
        sqrt_default_259 = None
        sub_tensor_259 = torch.ops.aten.sub.Tensor(
            convolution_default_479, mean_dim_259
        )
        mul_tensor_1813 = torch.ops.aten.mul.Tensor(
            sub_tensor_259, reciprocal_default_259
        )
        sub_tensor_259 = None
        squeeze_dim_1554 = torch.ops.aten.squeeze.dim(mean_dim_259, 3)
        mean_dim_259 = None
        squeeze_dim_1555 = torch.ops.aten.squeeze.dim(squeeze_dim_1554, 2)
        squeeze_dim_1554 = None
        squeeze_dim_1556 = torch.ops.aten.squeeze.dim(squeeze_dim_1555, 0)
        squeeze_dim_1555 = None
        squeeze_dim_1557 = torch.ops.aten.squeeze.dim(reciprocal_default_259, 3)
        reciprocal_default_259 = None
        squeeze_dim_1558 = torch.ops.aten.squeeze.dim(squeeze_dim_1557, 2)
        squeeze_dim_1557 = None
        squeeze_dim_1559 = torch.ops.aten.squeeze.dim(squeeze_dim_1558, 0)
        squeeze_dim_1558 = None
        unsqueeze_default_1036 = torch.ops.aten.unsqueeze.default(primals_997, -1)
        unsqueeze_default_1037 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1036, -1
        )
        unsqueeze_default_1036 = None
        unsqueeze_default_1038 = torch.ops.aten.unsqueeze.default(primals_998, -1)
        primals_998 = None
        unsqueeze_default_1039 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1038, -1
        )
        unsqueeze_default_1038 = None
        mul_tensor_1819 = torch.ops.aten.mul.Tensor(
            mul_tensor_1813, unsqueeze_default_1037
        )
        mul_tensor_1813 = unsqueeze_default_1037 = None
        add_tensor_1146 = torch.ops.aten.add.Tensor(
            mul_tensor_1819, unsqueeze_default_1039
        )
        mul_tensor_1819 = unsqueeze_default_1039 = None
        convolution_default_480 = torch.ops.aten.convolution.default(
            relu_default_255,
            primals_999,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_255 = None
        convolution_default_481 = torch.ops.aten.convolution.default(
            convolution_default_480,
            primals_1000,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_520 = torch.ops.aten.var.correction(
            convolution_default_481, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_260 = torch.ops.aten.mean.dim(convolution_default_481, [0, 2, 3], True)
        add_tensor_1147 = torch.ops.aten.add.Tensor(var_correction_520, 0.001)
        var_correction_520 = None
        sqrt_default_260 = torch.ops.aten.sqrt.default(add_tensor_1147)
        add_tensor_1147 = None
        reciprocal_default_260 = torch.ops.aten.reciprocal.default(sqrt_default_260)
        sqrt_default_260 = None
        sub_tensor_260 = torch.ops.aten.sub.Tensor(
            convolution_default_481, mean_dim_260
        )
        mul_tensor_1820 = torch.ops.aten.mul.Tensor(
            sub_tensor_260, reciprocal_default_260
        )
        sub_tensor_260 = None
        squeeze_dim_1560 = torch.ops.aten.squeeze.dim(mean_dim_260, 3)
        mean_dim_260 = None
        squeeze_dim_1561 = torch.ops.aten.squeeze.dim(squeeze_dim_1560, 2)
        squeeze_dim_1560 = None
        squeeze_dim_1562 = torch.ops.aten.squeeze.dim(squeeze_dim_1561, 0)
        squeeze_dim_1561 = None
        squeeze_dim_1563 = torch.ops.aten.squeeze.dim(reciprocal_default_260, 3)
        reciprocal_default_260 = None
        squeeze_dim_1564 = torch.ops.aten.squeeze.dim(squeeze_dim_1563, 2)
        squeeze_dim_1563 = None
        squeeze_dim_1565 = torch.ops.aten.squeeze.dim(squeeze_dim_1564, 0)
        squeeze_dim_1564 = None
        unsqueeze_default_1040 = torch.ops.aten.unsqueeze.default(primals_1001, -1)
        unsqueeze_default_1041 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1040, -1
        )
        unsqueeze_default_1040 = None
        unsqueeze_default_1042 = torch.ops.aten.unsqueeze.default(primals_1002, -1)
        primals_1002 = None
        unsqueeze_default_1043 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1042, -1
        )
        unsqueeze_default_1042 = None
        mul_tensor_1826 = torch.ops.aten.mul.Tensor(
            mul_tensor_1820, unsqueeze_default_1041
        )
        mul_tensor_1820 = unsqueeze_default_1041 = None
        add_tensor_1150 = torch.ops.aten.add.Tensor(
            mul_tensor_1826, unsqueeze_default_1043
        )
        mul_tensor_1826 = unsqueeze_default_1043 = None
        relu_default_260 = torch.ops.aten.relu.default(add_tensor_1150)
        add_tensor_1150 = None
        convolution_default_482 = torch.ops.aten.convolution.default(
            relu_default_260,
            primals_1003,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_483 = torch.ops.aten.convolution.default(
            convolution_default_482,
            primals_1004,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_522 = torch.ops.aten.var.correction(
            convolution_default_483, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_261 = torch.ops.aten.mean.dim(convolution_default_483, [0, 2, 3], True)
        add_tensor_1151 = torch.ops.aten.add.Tensor(var_correction_522, 0.001)
        var_correction_522 = None
        sqrt_default_261 = torch.ops.aten.sqrt.default(add_tensor_1151)
        add_tensor_1151 = None
        reciprocal_default_261 = torch.ops.aten.reciprocal.default(sqrt_default_261)
        sqrt_default_261 = None
        sub_tensor_261 = torch.ops.aten.sub.Tensor(
            convolution_default_483, mean_dim_261
        )
        mul_tensor_1827 = torch.ops.aten.mul.Tensor(
            sub_tensor_261, reciprocal_default_261
        )
        sub_tensor_261 = None
        squeeze_dim_1566 = torch.ops.aten.squeeze.dim(mean_dim_261, 3)
        mean_dim_261 = None
        squeeze_dim_1567 = torch.ops.aten.squeeze.dim(squeeze_dim_1566, 2)
        squeeze_dim_1566 = None
        squeeze_dim_1568 = torch.ops.aten.squeeze.dim(squeeze_dim_1567, 0)
        squeeze_dim_1567 = None
        squeeze_dim_1569 = torch.ops.aten.squeeze.dim(reciprocal_default_261, 3)
        reciprocal_default_261 = None
        squeeze_dim_1570 = torch.ops.aten.squeeze.dim(squeeze_dim_1569, 2)
        squeeze_dim_1569 = None
        squeeze_dim_1571 = torch.ops.aten.squeeze.dim(squeeze_dim_1570, 0)
        squeeze_dim_1570 = None
        unsqueeze_default_1044 = torch.ops.aten.unsqueeze.default(primals_1005, -1)
        unsqueeze_default_1045 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1044, -1
        )
        unsqueeze_default_1044 = None
        unsqueeze_default_1046 = torch.ops.aten.unsqueeze.default(primals_1006, -1)
        primals_1006 = None
        unsqueeze_default_1047 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1046, -1
        )
        unsqueeze_default_1046 = None
        mul_tensor_1833 = torch.ops.aten.mul.Tensor(
            mul_tensor_1827, unsqueeze_default_1045
        )
        mul_tensor_1827 = unsqueeze_default_1045 = None
        add_tensor_1154 = torch.ops.aten.add.Tensor(
            mul_tensor_1833, unsqueeze_default_1047
        )
        mul_tensor_1833 = unsqueeze_default_1047 = None
        add_tensor_1155 = torch.ops.aten.add.Tensor(add_tensor_1146, add_tensor_1154)
        add_tensor_1146 = add_tensor_1154 = None
        avg_pool2d_default_67 = torch.ops.aten.avg_pool2d.default(
            add_tensor_1121, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_1156 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_67, add_tensor_1117
        )
        avg_pool2d_default_67 = None
        avg_pool2d_default_68 = torch.ops.aten.avg_pool2d.default(
            add_tensor_1117, [3, 3], [1, 1], [1, 1], False, False
        )
        add_tensor_1157 = torch.ops.aten.add.Tensor(
            avg_pool2d_default_68, avg_pool2d_default_68
        )
        avg_pool2d_default_68 = None
        convolution_default_484 = torch.ops.aten.convolution.default(
            relu_default_253,
            primals_1007,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        relu_default_253 = None
        convolution_default_485 = torch.ops.aten.convolution.default(
            convolution_default_484,
            primals_1008,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_524 = torch.ops.aten.var.correction(
            convolution_default_485, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_262 = torch.ops.aten.mean.dim(convolution_default_485, [0, 2, 3], True)
        add_tensor_1158 = torch.ops.aten.add.Tensor(var_correction_524, 0.001)
        var_correction_524 = None
        sqrt_default_262 = torch.ops.aten.sqrt.default(add_tensor_1158)
        add_tensor_1158 = None
        reciprocal_default_262 = torch.ops.aten.reciprocal.default(sqrt_default_262)
        sqrt_default_262 = None
        sub_tensor_262 = torch.ops.aten.sub.Tensor(
            convolution_default_485, mean_dim_262
        )
        mul_tensor_1834 = torch.ops.aten.mul.Tensor(
            sub_tensor_262, reciprocal_default_262
        )
        sub_tensor_262 = None
        squeeze_dim_1572 = torch.ops.aten.squeeze.dim(mean_dim_262, 3)
        mean_dim_262 = None
        squeeze_dim_1573 = torch.ops.aten.squeeze.dim(squeeze_dim_1572, 2)
        squeeze_dim_1572 = None
        squeeze_dim_1574 = torch.ops.aten.squeeze.dim(squeeze_dim_1573, 0)
        squeeze_dim_1573 = None
        squeeze_dim_1575 = torch.ops.aten.squeeze.dim(reciprocal_default_262, 3)
        reciprocal_default_262 = None
        squeeze_dim_1576 = torch.ops.aten.squeeze.dim(squeeze_dim_1575, 2)
        squeeze_dim_1575 = None
        squeeze_dim_1577 = torch.ops.aten.squeeze.dim(squeeze_dim_1576, 0)
        squeeze_dim_1576 = None
        unsqueeze_default_1048 = torch.ops.aten.unsqueeze.default(primals_1009, -1)
        unsqueeze_default_1049 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1048, -1
        )
        unsqueeze_default_1048 = None
        unsqueeze_default_1050 = torch.ops.aten.unsqueeze.default(primals_1010, -1)
        primals_1010 = None
        unsqueeze_default_1051 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1050, -1
        )
        unsqueeze_default_1050 = None
        mul_tensor_1840 = torch.ops.aten.mul.Tensor(
            mul_tensor_1834, unsqueeze_default_1049
        )
        mul_tensor_1834 = unsqueeze_default_1049 = None
        add_tensor_1161 = torch.ops.aten.add.Tensor(
            mul_tensor_1840, unsqueeze_default_1051
        )
        mul_tensor_1840 = unsqueeze_default_1051 = None
        relu_default_262 = torch.ops.aten.relu.default(add_tensor_1161)
        add_tensor_1161 = None
        convolution_default_486 = torch.ops.aten.convolution.default(
            relu_default_262,
            primals_1011,
            None,
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            672,
        )
        convolution_default_487 = torch.ops.aten.convolution.default(
            convolution_default_486,
            primals_1012,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        var_correction_526 = torch.ops.aten.var.correction(
            convolution_default_487, [0, 2, 3], correction=0, keepdim=True
        )
        mean_dim_263 = torch.ops.aten.mean.dim(convolution_default_487, [0, 2, 3], True)
        add_tensor_1162 = torch.ops.aten.add.Tensor(var_correction_526, 0.001)
        var_correction_526 = None
        sqrt_default_263 = torch.ops.aten.sqrt.default(add_tensor_1162)
        add_tensor_1162 = None
        reciprocal_default_263 = torch.ops.aten.reciprocal.default(sqrt_default_263)
        sqrt_default_263 = None
        sub_tensor_263 = torch.ops.aten.sub.Tensor(
            convolution_default_487, mean_dim_263
        )
        mul_tensor_1841 = torch.ops.aten.mul.Tensor(
            sub_tensor_263, reciprocal_default_263
        )
        sub_tensor_263 = None
        squeeze_dim_1578 = torch.ops.aten.squeeze.dim(mean_dim_263, 3)
        mean_dim_263 = None
        squeeze_dim_1579 = torch.ops.aten.squeeze.dim(squeeze_dim_1578, 2)
        squeeze_dim_1578 = None
        squeeze_dim_1580 = torch.ops.aten.squeeze.dim(squeeze_dim_1579, 0)
        squeeze_dim_1579 = None
        squeeze_dim_1581 = torch.ops.aten.squeeze.dim(reciprocal_default_263, 3)
        reciprocal_default_263 = None
        squeeze_dim_1582 = torch.ops.aten.squeeze.dim(squeeze_dim_1581, 2)
        squeeze_dim_1581 = None
        squeeze_dim_1583 = torch.ops.aten.squeeze.dim(squeeze_dim_1582, 0)
        squeeze_dim_1582 = None
        unsqueeze_default_1052 = torch.ops.aten.unsqueeze.default(primals_1013, -1)
        unsqueeze_default_1053 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1052, -1
        )
        unsqueeze_default_1052 = None
        unsqueeze_default_1054 = torch.ops.aten.unsqueeze.default(primals_1014, -1)
        primals_1014 = None
        unsqueeze_default_1055 = torch.ops.aten.unsqueeze.default(
            unsqueeze_default_1054, -1
        )
        unsqueeze_default_1054 = None
        mul_tensor_1847 = torch.ops.aten.mul.Tensor(
            mul_tensor_1841, unsqueeze_default_1053
        )
        mul_tensor_1841 = unsqueeze_default_1053 = None
        add_tensor_1165 = torch.ops.aten.add.Tensor(
            mul_tensor_1847, unsqueeze_default_1055
        )
        mul_tensor_1847 = unsqueeze_default_1055 = None
        add_tensor_1166 = torch.ops.aten.add.Tensor(add_tensor_1165, add_tensor_1121)
        add_tensor_1165 = add_tensor_1121 = None
        cat_default_25 = torch.ops.aten.cat.default(
            [
                add_tensor_1117,
                add_tensor_1138,
                add_tensor_1155,
                add_tensor_1156,
                add_tensor_1157,
                add_tensor_1166,
            ],
            1,
        )
        add_tensor_1117 = (
            add_tensor_1138
        ) = add_tensor_1155 = add_tensor_1156 = add_tensor_1157 = add_tensor_1166 = None
        relu_default_263 = torch.ops.aten.relu.default(cat_default_25)
        cat_default_25 = None
        mean_dim_264 = torch.ops.aten.mean.dim(relu_default_263, [-1, -2], True)
        view_default = torch.ops.aten.view.default(mean_dim_264, [16, 4032])
        mean_dim_264 = None
        permute_default = torch.ops.aten.permute.default(primals_1015, [1, 0])
        primals_1015 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default)
        add_tensor_1167 = torch.ops.aten.add.Tensor(mm_default, primals_1016)
        mm_default = primals_1016 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [1, 0])
        permute_default = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default_263, 0)
        relu_default_263 = None
        view_default_2 = torch.ops.aten.view.default(squeeze_dim_1580, [1, 672, 1, 1])
        squeeze_dim_1580 = None
        view_default_6 = torch.ops.aten.view.default(squeeze_dim_1574, [1, 672, 1, 1])
        squeeze_dim_1574 = None
        view_default_10 = torch.ops.aten.view.default(squeeze_dim_1568, [1, 672, 1, 1])
        squeeze_dim_1568 = None
        view_default_14 = torch.ops.aten.view.default(squeeze_dim_1562, [1, 672, 1, 1])
        squeeze_dim_1562 = None
        view_default_18 = torch.ops.aten.view.default(squeeze_dim_1556, [1, 672, 1, 1])
        squeeze_dim_1556 = None
        view_default_22 = torch.ops.aten.view.default(squeeze_dim_1550, [1, 672, 1, 1])
        squeeze_dim_1550 = None
        view_default_26 = torch.ops.aten.view.default(squeeze_dim_1544, [1, 672, 1, 1])
        squeeze_dim_1544 = None
        view_default_30 = torch.ops.aten.view.default(squeeze_dim_1538, [1, 672, 1, 1])
        squeeze_dim_1538 = None
        view_default_34 = torch.ops.aten.view.default(squeeze_dim_1532, [1, 672, 1, 1])
        squeeze_dim_1532 = None
        view_default_38 = torch.ops.aten.view.default(squeeze_dim_1526, [1, 672, 1, 1])
        squeeze_dim_1526 = None
        view_default_50 = torch.ops.aten.view.default(squeeze_dim_1508, [1, 672, 1, 1])
        squeeze_dim_1508 = None
        view_default_54 = torch.ops.aten.view.default(squeeze_dim_1502, [1, 672, 1, 1])
        squeeze_dim_1502 = None
        view_default_58 = torch.ops.aten.view.default(squeeze_dim_1496, [1, 672, 1, 1])
        squeeze_dim_1496 = None
        view_default_62 = torch.ops.aten.view.default(squeeze_dim_1490, [1, 672, 1, 1])
        squeeze_dim_1490 = None
        view_default_66 = torch.ops.aten.view.default(squeeze_dim_1484, [1, 672, 1, 1])
        squeeze_dim_1484 = None
        view_default_70 = torch.ops.aten.view.default(squeeze_dim_1478, [1, 672, 1, 1])
        squeeze_dim_1478 = None
        view_default_74 = torch.ops.aten.view.default(squeeze_dim_1472, [1, 672, 1, 1])
        squeeze_dim_1472 = None
        view_default_78 = torch.ops.aten.view.default(squeeze_dim_1466, [1, 672, 1, 1])
        squeeze_dim_1466 = None
        view_default_82 = torch.ops.aten.view.default(squeeze_dim_1460, [1, 672, 1, 1])
        squeeze_dim_1460 = None
        view_default_86 = torch.ops.aten.view.default(squeeze_dim_1454, [1, 672, 1, 1])
        squeeze_dim_1454 = None
        view_default_98 = torch.ops.aten.view.default(squeeze_dim_1436, [1, 672, 1, 1])
        squeeze_dim_1436 = None
        view_default_102 = torch.ops.aten.view.default(squeeze_dim_1430, [1, 672, 1, 1])
        squeeze_dim_1430 = None
        view_default_106 = torch.ops.aten.view.default(squeeze_dim_1424, [1, 672, 1, 1])
        squeeze_dim_1424 = None
        view_default_110 = torch.ops.aten.view.default(squeeze_dim_1418, [1, 672, 1, 1])
        squeeze_dim_1418 = None
        view_default_114 = torch.ops.aten.view.default(squeeze_dim_1412, [1, 672, 1, 1])
        squeeze_dim_1412 = None
        view_default_118 = torch.ops.aten.view.default(squeeze_dim_1406, [1, 672, 1, 1])
        squeeze_dim_1406 = None
        view_default_122 = torch.ops.aten.view.default(squeeze_dim_1400, [1, 672, 1, 1])
        squeeze_dim_1400 = None
        view_default_126 = torch.ops.aten.view.default(squeeze_dim_1394, [1, 672, 1, 1])
        squeeze_dim_1394 = None
        view_default_130 = torch.ops.aten.view.default(squeeze_dim_1388, [1, 672, 1, 1])
        squeeze_dim_1388 = None
        view_default_134 = torch.ops.aten.view.default(squeeze_dim_1382, [1, 672, 1, 1])
        squeeze_dim_1382 = None
        view_default_146 = torch.ops.aten.view.default(squeeze_dim_1364, [1, 672, 1, 1])
        squeeze_dim_1364 = None
        view_default_150 = torch.ops.aten.view.default(squeeze_dim_1358, [1, 672, 1, 1])
        squeeze_dim_1358 = None
        view_default_154 = torch.ops.aten.view.default(squeeze_dim_1352, [1, 672, 1, 1])
        squeeze_dim_1352 = None
        view_default_158 = torch.ops.aten.view.default(squeeze_dim_1346, [1, 672, 1, 1])
        squeeze_dim_1346 = None
        view_default_162 = torch.ops.aten.view.default(squeeze_dim_1340, [1, 672, 1, 1])
        squeeze_dim_1340 = None
        view_default_166 = torch.ops.aten.view.default(squeeze_dim_1334, [1, 672, 1, 1])
        squeeze_dim_1334 = None
        view_default_170 = torch.ops.aten.view.default(squeeze_dim_1328, [1, 672, 1, 1])
        squeeze_dim_1328 = None
        view_default_174 = torch.ops.aten.view.default(squeeze_dim_1322, [1, 672, 1, 1])
        squeeze_dim_1322 = None
        view_default_178 = torch.ops.aten.view.default(squeeze_dim_1316, [1, 672, 1, 1])
        squeeze_dim_1316 = None
        view_default_182 = torch.ops.aten.view.default(squeeze_dim_1310, [1, 672, 1, 1])
        squeeze_dim_1310 = None
        view_default_194 = torch.ops.aten.view.default(squeeze_dim_1292, [1, 672, 1, 1])
        squeeze_dim_1292 = None
        view_default_198 = torch.ops.aten.view.default(squeeze_dim_1286, [1, 672, 1, 1])
        squeeze_dim_1286 = None
        view_default_202 = torch.ops.aten.view.default(squeeze_dim_1280, [1, 672, 1, 1])
        squeeze_dim_1280 = None
        view_default_206 = torch.ops.aten.view.default(squeeze_dim_1274, [1, 672, 1, 1])
        squeeze_dim_1274 = None
        view_default_210 = torch.ops.aten.view.default(squeeze_dim_1268, [1, 672, 1, 1])
        squeeze_dim_1268 = None
        view_default_214 = torch.ops.aten.view.default(squeeze_dim_1262, [1, 672, 1, 1])
        squeeze_dim_1262 = None
        view_default_218 = torch.ops.aten.view.default(squeeze_dim_1256, [1, 672, 1, 1])
        squeeze_dim_1256 = None
        view_default_222 = torch.ops.aten.view.default(squeeze_dim_1250, [1, 672, 1, 1])
        squeeze_dim_1250 = None
        view_default_226 = torch.ops.aten.view.default(squeeze_dim_1244, [1, 672, 1, 1])
        squeeze_dim_1244 = None
        view_default_230 = torch.ops.aten.view.default(squeeze_dim_1238, [1, 672, 1, 1])
        squeeze_dim_1238 = None
        view_default_242 = torch.ops.aten.view.default(squeeze_dim_1220, [1, 672, 1, 1])
        squeeze_dim_1220 = None
        view_default_246 = torch.ops.aten.view.default(squeeze_dim_1214, [1, 672, 1, 1])
        squeeze_dim_1214 = None
        view_default_250 = torch.ops.aten.view.default(squeeze_dim_1208, [1, 672, 1, 1])
        squeeze_dim_1208 = None
        view_default_254 = torch.ops.aten.view.default(squeeze_dim_1202, [1, 672, 1, 1])
        squeeze_dim_1202 = None
        view_default_258 = torch.ops.aten.view.default(squeeze_dim_1196, [1, 672, 1, 1])
        squeeze_dim_1196 = None
        view_default_262 = torch.ops.aten.view.default(squeeze_dim_1190, [1, 672, 1, 1])
        squeeze_dim_1190 = None
        view_default_266 = torch.ops.aten.view.default(squeeze_dim_1184, [1, 672, 1, 1])
        squeeze_dim_1184 = None
        view_default_270 = torch.ops.aten.view.default(squeeze_dim_1178, [1, 672, 1, 1])
        squeeze_dim_1178 = None
        view_default_274 = torch.ops.aten.view.default(squeeze_dim_1172, [1, 672, 1, 1])
        squeeze_dim_1172 = None
        view_default_278 = torch.ops.aten.view.default(squeeze_dim_1166, [1, 672, 1, 1])
        squeeze_dim_1166 = None
        view_default_290 = torch.ops.aten.view.default(squeeze_dim_1148, [1, 672, 1, 1])
        squeeze_dim_1148 = None
        view_default_294 = torch.ops.aten.view.default(squeeze_dim_1142, [1, 672, 1, 1])
        squeeze_dim_1142 = None
        view_default_298 = torch.ops.aten.view.default(squeeze_dim_1136, [1, 672, 1, 1])
        squeeze_dim_1136 = None
        view_default_302 = torch.ops.aten.view.default(squeeze_dim_1130, [1, 672, 1, 1])
        squeeze_dim_1130 = None
        le_scalar_76 = torch.ops.aten.le.Scalar(relu_default_183, 0)
        relu_default_183 = None
        view_default_306 = torch.ops.aten.view.default(squeeze_dim_1124, [1, 672, 1, 1])
        squeeze_dim_1124 = None
        view_default_310 = torch.ops.aten.view.default(squeeze_dim_1118, [1, 672, 1, 1])
        squeeze_dim_1118 = None
        view_default_314 = torch.ops.aten.view.default(squeeze_dim_1112, [1, 672, 1, 1])
        squeeze_dim_1112 = None
        view_default_318 = torch.ops.aten.view.default(squeeze_dim_1106, [1, 672, 1, 1])
        squeeze_dim_1106 = None
        view_default_322 = torch.ops.aten.view.default(squeeze_dim_1100, [1, 672, 1, 1])
        squeeze_dim_1100 = None
        view_default_326 = torch.ops.aten.view.default(squeeze_dim_1094, [1, 672, 1, 1])
        squeeze_dim_1094 = None
        le_scalar_82 = torch.ops.aten.le.Scalar(relu_default_181, 0)
        relu_default_181 = None
        view_default_330 = torch.ops.aten.view.default(squeeze_dim_1088, [1, 672, 1, 1])
        squeeze_dim_1088 = None
        view_default_334 = torch.ops.aten.view.default(squeeze_dim_1082, [1, 672, 1, 1])
        squeeze_dim_1082 = None
        view_default_338 = torch.ops.aten.view.default(squeeze_dim_1076, [1, 336, 1, 1])
        squeeze_dim_1076 = None
        view_default_342 = torch.ops.aten.view.default(squeeze_dim_1070, [1, 336, 1, 1])
        squeeze_dim_1070 = None
        view_default_346 = torch.ops.aten.view.default(squeeze_dim_1064, [1, 336, 1, 1])
        squeeze_dim_1064 = None
        view_default_350 = torch.ops.aten.view.default(squeeze_dim_1058, [1, 336, 1, 1])
        squeeze_dim_1058 = None
        view_default_354 = torch.ops.aten.view.default(squeeze_dim_1052, [1, 336, 1, 1])
        squeeze_dim_1052 = None
        view_default_358 = torch.ops.aten.view.default(squeeze_dim_1046, [1, 336, 1, 1])
        squeeze_dim_1046 = None
        view_default_362 = torch.ops.aten.view.default(squeeze_dim_1040, [1, 336, 1, 1])
        squeeze_dim_1040 = None
        view_default_366 = torch.ops.aten.view.default(squeeze_dim_1034, [1, 336, 1, 1])
        squeeze_dim_1034 = None
        view_default_370 = torch.ops.aten.view.default(squeeze_dim_1028, [1, 336, 1, 1])
        squeeze_dim_1028 = None
        view_default_374 = torch.ops.aten.view.default(squeeze_dim_1022, [1, 336, 1, 1])
        squeeze_dim_1022 = None
        view_default_386 = torch.ops.aten.view.default(squeeze_dim_1004, [1, 336, 1, 1])
        squeeze_dim_1004 = None
        view_default_390 = torch.ops.aten.view.default(squeeze_dim_998, [1, 336, 1, 1])
        squeeze_dim_998 = None
        view_default_394 = torch.ops.aten.view.default(squeeze_dim_992, [1, 336, 1, 1])
        squeeze_dim_992 = None
        view_default_398 = torch.ops.aten.view.default(squeeze_dim_986, [1, 336, 1, 1])
        squeeze_dim_986 = None
        view_default_402 = torch.ops.aten.view.default(squeeze_dim_980, [1, 336, 1, 1])
        squeeze_dim_980 = None
        view_default_406 = torch.ops.aten.view.default(squeeze_dim_974, [1, 336, 1, 1])
        squeeze_dim_974 = None
        view_default_410 = torch.ops.aten.view.default(squeeze_dim_968, [1, 336, 1, 1])
        squeeze_dim_968 = None
        view_default_414 = torch.ops.aten.view.default(squeeze_dim_962, [1, 336, 1, 1])
        squeeze_dim_962 = None
        view_default_418 = torch.ops.aten.view.default(squeeze_dim_956, [1, 336, 1, 1])
        squeeze_dim_956 = None
        view_default_422 = torch.ops.aten.view.default(squeeze_dim_950, [1, 336, 1, 1])
        squeeze_dim_950 = None
        view_default_434 = torch.ops.aten.view.default(squeeze_dim_932, [1, 336, 1, 1])
        squeeze_dim_932 = None
        view_default_438 = torch.ops.aten.view.default(squeeze_dim_926, [1, 336, 1, 1])
        squeeze_dim_926 = None
        view_default_442 = torch.ops.aten.view.default(squeeze_dim_920, [1, 336, 1, 1])
        squeeze_dim_920 = None
        view_default_446 = torch.ops.aten.view.default(squeeze_dim_914, [1, 336, 1, 1])
        squeeze_dim_914 = None
        view_default_450 = torch.ops.aten.view.default(squeeze_dim_908, [1, 336, 1, 1])
        squeeze_dim_908 = None
        view_default_454 = torch.ops.aten.view.default(squeeze_dim_902, [1, 336, 1, 1])
        squeeze_dim_902 = None
        view_default_458 = torch.ops.aten.view.default(squeeze_dim_896, [1, 336, 1, 1])
        squeeze_dim_896 = None
        view_default_462 = torch.ops.aten.view.default(squeeze_dim_890, [1, 336, 1, 1])
        squeeze_dim_890 = None
        view_default_466 = torch.ops.aten.view.default(squeeze_dim_884, [1, 336, 1, 1])
        squeeze_dim_884 = None
        view_default_470 = torch.ops.aten.view.default(squeeze_dim_878, [1, 336, 1, 1])
        squeeze_dim_878 = None
        view_default_482 = torch.ops.aten.view.default(squeeze_dim_860, [1, 336, 1, 1])
        squeeze_dim_860 = None
        view_default_486 = torch.ops.aten.view.default(squeeze_dim_854, [1, 336, 1, 1])
        squeeze_dim_854 = None
        view_default_490 = torch.ops.aten.view.default(squeeze_dim_848, [1, 336, 1, 1])
        squeeze_dim_848 = None
        view_default_494 = torch.ops.aten.view.default(squeeze_dim_842, [1, 336, 1, 1])
        squeeze_dim_842 = None
        view_default_498 = torch.ops.aten.view.default(squeeze_dim_836, [1, 336, 1, 1])
        squeeze_dim_836 = None
        view_default_502 = torch.ops.aten.view.default(squeeze_dim_830, [1, 336, 1, 1])
        squeeze_dim_830 = None
        view_default_506 = torch.ops.aten.view.default(squeeze_dim_824, [1, 336, 1, 1])
        squeeze_dim_824 = None
        view_default_510 = torch.ops.aten.view.default(squeeze_dim_818, [1, 336, 1, 1])
        squeeze_dim_818 = None
        view_default_514 = torch.ops.aten.view.default(squeeze_dim_812, [1, 336, 1, 1])
        squeeze_dim_812 = None
        view_default_518 = torch.ops.aten.view.default(squeeze_dim_806, [1, 336, 1, 1])
        squeeze_dim_806 = None
        view_default_530 = torch.ops.aten.view.default(squeeze_dim_788, [1, 336, 1, 1])
        squeeze_dim_788 = None
        view_default_534 = torch.ops.aten.view.default(squeeze_dim_782, [1, 336, 1, 1])
        squeeze_dim_782 = None
        view_default_538 = torch.ops.aten.view.default(squeeze_dim_776, [1, 336, 1, 1])
        squeeze_dim_776 = None
        view_default_542 = torch.ops.aten.view.default(squeeze_dim_770, [1, 336, 1, 1])
        squeeze_dim_770 = None
        view_default_546 = torch.ops.aten.view.default(squeeze_dim_764, [1, 336, 1, 1])
        squeeze_dim_764 = None
        view_default_550 = torch.ops.aten.view.default(squeeze_dim_758, [1, 336, 1, 1])
        squeeze_dim_758 = None
        view_default_554 = torch.ops.aten.view.default(squeeze_dim_752, [1, 336, 1, 1])
        squeeze_dim_752 = None
        view_default_558 = torch.ops.aten.view.default(squeeze_dim_746, [1, 336, 1, 1])
        squeeze_dim_746 = None
        view_default_562 = torch.ops.aten.view.default(squeeze_dim_740, [1, 336, 1, 1])
        squeeze_dim_740 = None
        view_default_566 = torch.ops.aten.view.default(squeeze_dim_734, [1, 336, 1, 1])
        squeeze_dim_734 = None
        view_default_578 = torch.ops.aten.view.default(squeeze_dim_716, [1, 336, 1, 1])
        squeeze_dim_716 = None
        view_default_582 = torch.ops.aten.view.default(squeeze_dim_710, [1, 336, 1, 1])
        squeeze_dim_710 = None
        view_default_586 = torch.ops.aten.view.default(squeeze_dim_704, [1, 336, 1, 1])
        squeeze_dim_704 = None
        view_default_590 = torch.ops.aten.view.default(squeeze_dim_698, [1, 336, 1, 1])
        squeeze_dim_698 = None
        view_default_594 = torch.ops.aten.view.default(squeeze_dim_692, [1, 336, 1, 1])
        squeeze_dim_692 = None
        view_default_598 = torch.ops.aten.view.default(squeeze_dim_686, [1, 336, 1, 1])
        squeeze_dim_686 = None
        view_default_602 = torch.ops.aten.view.default(squeeze_dim_680, [1, 336, 1, 1])
        squeeze_dim_680 = None
        view_default_606 = torch.ops.aten.view.default(squeeze_dim_674, [1, 336, 1, 1])
        squeeze_dim_674 = None
        view_default_610 = torch.ops.aten.view.default(squeeze_dim_668, [1, 336, 1, 1])
        squeeze_dim_668 = None
        view_default_614 = torch.ops.aten.view.default(squeeze_dim_662, [1, 336, 1, 1])
        squeeze_dim_662 = None
        view_default_626 = torch.ops.aten.view.default(squeeze_dim_644, [1, 336, 1, 1])
        squeeze_dim_644 = None
        view_default_630 = torch.ops.aten.view.default(squeeze_dim_638, [1, 336, 1, 1])
        squeeze_dim_638 = None
        view_default_634 = torch.ops.aten.view.default(squeeze_dim_632, [1, 336, 1, 1])
        squeeze_dim_632 = None
        view_default_638 = torch.ops.aten.view.default(squeeze_dim_626, [1, 336, 1, 1])
        squeeze_dim_626 = None
        le_scalar_160 = torch.ops.aten.le.Scalar(relu_default_99, 0)
        relu_default_99 = None
        view_default_642 = torch.ops.aten.view.default(squeeze_dim_620, [1, 336, 1, 1])
        squeeze_dim_620 = None
        view_default_646 = torch.ops.aten.view.default(squeeze_dim_614, [1, 336, 1, 1])
        squeeze_dim_614 = None
        view_default_650 = torch.ops.aten.view.default(squeeze_dim_608, [1, 336, 1, 1])
        squeeze_dim_608 = None
        view_default_654 = torch.ops.aten.view.default(squeeze_dim_602, [1, 336, 1, 1])
        squeeze_dim_602 = None
        view_default_658 = torch.ops.aten.view.default(squeeze_dim_596, [1, 336, 1, 1])
        squeeze_dim_596 = None
        view_default_662 = torch.ops.aten.view.default(squeeze_dim_590, [1, 336, 1, 1])
        squeeze_dim_590 = None
        le_scalar_166 = torch.ops.aten.le.Scalar(relu_default_97, 0)
        relu_default_97 = None
        view_default_666 = torch.ops.aten.view.default(squeeze_dim_584, [1, 336, 1, 1])
        squeeze_dim_584 = None
        view_default_670 = torch.ops.aten.view.default(squeeze_dim_578, [1, 336, 1, 1])
        squeeze_dim_578 = None
        view_default_674 = torch.ops.aten.view.default(squeeze_dim_572, [1, 168, 1, 1])
        squeeze_dim_572 = None
        view_default_678 = torch.ops.aten.view.default(squeeze_dim_566, [1, 168, 1, 1])
        squeeze_dim_566 = None
        view_default_682 = torch.ops.aten.view.default(squeeze_dim_560, [1, 168, 1, 1])
        squeeze_dim_560 = None
        view_default_686 = torch.ops.aten.view.default(squeeze_dim_554, [1, 168, 1, 1])
        squeeze_dim_554 = None
        view_default_690 = torch.ops.aten.view.default(squeeze_dim_548, [1, 168, 1, 1])
        squeeze_dim_548 = None
        view_default_694 = torch.ops.aten.view.default(squeeze_dim_542, [1, 168, 1, 1])
        squeeze_dim_542 = None
        view_default_698 = torch.ops.aten.view.default(squeeze_dim_536, [1, 168, 1, 1])
        squeeze_dim_536 = None
        view_default_702 = torch.ops.aten.view.default(squeeze_dim_530, [1, 168, 1, 1])
        squeeze_dim_530 = None
        view_default_706 = torch.ops.aten.view.default(squeeze_dim_524, [1, 168, 1, 1])
        squeeze_dim_524 = None
        view_default_710 = torch.ops.aten.view.default(squeeze_dim_518, [1, 168, 1, 1])
        squeeze_dim_518 = None
        view_default_722 = torch.ops.aten.view.default(squeeze_dim_500, [1, 168, 1, 1])
        squeeze_dim_500 = None
        view_default_726 = torch.ops.aten.view.default(squeeze_dim_494, [1, 168, 1, 1])
        squeeze_dim_494 = None
        view_default_730 = torch.ops.aten.view.default(squeeze_dim_488, [1, 168, 1, 1])
        squeeze_dim_488 = None
        view_default_734 = torch.ops.aten.view.default(squeeze_dim_482, [1, 168, 1, 1])
        squeeze_dim_482 = None
        view_default_738 = torch.ops.aten.view.default(squeeze_dim_476, [1, 168, 1, 1])
        squeeze_dim_476 = None
        view_default_742 = torch.ops.aten.view.default(squeeze_dim_470, [1, 168, 1, 1])
        squeeze_dim_470 = None
        view_default_746 = torch.ops.aten.view.default(squeeze_dim_464, [1, 168, 1, 1])
        squeeze_dim_464 = None
        view_default_750 = torch.ops.aten.view.default(squeeze_dim_458, [1, 168, 1, 1])
        squeeze_dim_458 = None
        view_default_754 = torch.ops.aten.view.default(squeeze_dim_452, [1, 168, 1, 1])
        squeeze_dim_452 = None
        view_default_758 = torch.ops.aten.view.default(squeeze_dim_446, [1, 168, 1, 1])
        squeeze_dim_446 = None
        view_default_770 = torch.ops.aten.view.default(squeeze_dim_428, [1, 168, 1, 1])
        squeeze_dim_428 = None
        view_default_774 = torch.ops.aten.view.default(squeeze_dim_422, [1, 168, 1, 1])
        squeeze_dim_422 = None
        view_default_778 = torch.ops.aten.view.default(squeeze_dim_416, [1, 168, 1, 1])
        squeeze_dim_416 = None
        view_default_782 = torch.ops.aten.view.default(squeeze_dim_410, [1, 168, 1, 1])
        squeeze_dim_410 = None
        view_default_786 = torch.ops.aten.view.default(squeeze_dim_404, [1, 168, 1, 1])
        squeeze_dim_404 = None
        view_default_790 = torch.ops.aten.view.default(squeeze_dim_398, [1, 168, 1, 1])
        squeeze_dim_398 = None
        view_default_794 = torch.ops.aten.view.default(squeeze_dim_392, [1, 168, 1, 1])
        squeeze_dim_392 = None
        view_default_798 = torch.ops.aten.view.default(squeeze_dim_386, [1, 168, 1, 1])
        squeeze_dim_386 = None
        view_default_802 = torch.ops.aten.view.default(squeeze_dim_380, [1, 168, 1, 1])
        squeeze_dim_380 = None
        view_default_806 = torch.ops.aten.view.default(squeeze_dim_374, [1, 168, 1, 1])
        squeeze_dim_374 = None
        view_default_818 = torch.ops.aten.view.default(squeeze_dim_356, [1, 168, 1, 1])
        squeeze_dim_356 = None
        view_default_822 = torch.ops.aten.view.default(squeeze_dim_350, [1, 168, 1, 1])
        squeeze_dim_350 = None
        view_default_826 = torch.ops.aten.view.default(squeeze_dim_344, [1, 168, 1, 1])
        squeeze_dim_344 = None
        view_default_830 = torch.ops.aten.view.default(squeeze_dim_338, [1, 168, 1, 1])
        squeeze_dim_338 = None
        view_default_834 = torch.ops.aten.view.default(squeeze_dim_332, [1, 168, 1, 1])
        squeeze_dim_332 = None
        view_default_838 = torch.ops.aten.view.default(squeeze_dim_326, [1, 168, 1, 1])
        squeeze_dim_326 = None
        view_default_842 = torch.ops.aten.view.default(squeeze_dim_320, [1, 168, 1, 1])
        squeeze_dim_320 = None
        view_default_846 = torch.ops.aten.view.default(squeeze_dim_314, [1, 168, 1, 1])
        squeeze_dim_314 = None
        view_default_850 = torch.ops.aten.view.default(squeeze_dim_308, [1, 168, 1, 1])
        squeeze_dim_308 = None
        view_default_854 = torch.ops.aten.view.default(squeeze_dim_302, [1, 168, 1, 1])
        squeeze_dim_302 = None
        view_default_866 = torch.ops.aten.view.default(squeeze_dim_284, [1, 168, 1, 1])
        squeeze_dim_284 = None
        view_default_870 = torch.ops.aten.view.default(squeeze_dim_278, [1, 168, 1, 1])
        squeeze_dim_278 = None
        view_default_874 = torch.ops.aten.view.default(squeeze_dim_272, [1, 168, 1, 1])
        squeeze_dim_272 = None
        view_default_878 = torch.ops.aten.view.default(squeeze_dim_266, [1, 168, 1, 1])
        squeeze_dim_266 = None
        view_default_882 = torch.ops.aten.view.default(squeeze_dim_260, [1, 168, 1, 1])
        squeeze_dim_260 = None
        view_default_886 = torch.ops.aten.view.default(squeeze_dim_254, [1, 168, 1, 1])
        squeeze_dim_254 = None
        view_default_890 = torch.ops.aten.view.default(squeeze_dim_248, [1, 168, 1, 1])
        squeeze_dim_248 = None
        view_default_894 = torch.ops.aten.view.default(squeeze_dim_242, [1, 168, 1, 1])
        squeeze_dim_242 = None
        view_default_898 = torch.ops.aten.view.default(squeeze_dim_236, [1, 168, 1, 1])
        squeeze_dim_236 = None
        view_default_902 = torch.ops.aten.view.default(squeeze_dim_230, [1, 168, 1, 1])
        squeeze_dim_230 = None
        view_default_914 = torch.ops.aten.view.default(squeeze_dim_212, [1, 168, 1, 1])
        squeeze_dim_212 = None
        view_default_918 = torch.ops.aten.view.default(squeeze_dim_206, [1, 168, 1, 1])
        squeeze_dim_206 = None
        view_default_922 = torch.ops.aten.view.default(squeeze_dim_200, [1, 168, 1, 1])
        squeeze_dim_200 = None
        view_default_926 = torch.ops.aten.view.default(squeeze_dim_194, [1, 168, 1, 1])
        squeeze_dim_194 = None
        view_default_930 = torch.ops.aten.view.default(squeeze_dim_188, [1, 168, 1, 1])
        squeeze_dim_188 = None
        view_default_934 = torch.ops.aten.view.default(squeeze_dim_182, [1, 168, 1, 1])
        squeeze_dim_182 = None
        view_default_938 = torch.ops.aten.view.default(squeeze_dim_176, [1, 168, 1, 1])
        squeeze_dim_176 = None
        view_default_942 = torch.ops.aten.view.default(squeeze_dim_170, [1, 168, 1, 1])
        squeeze_dim_170 = None
        view_default_946 = torch.ops.aten.view.default(squeeze_dim_164, [1, 168, 1, 1])
        squeeze_dim_164 = None
        view_default_950 = torch.ops.aten.view.default(squeeze_dim_158, [1, 168, 1, 1])
        squeeze_dim_158 = None
        view_default_962 = torch.ops.aten.view.default(squeeze_dim_140, [1, 84, 1, 1])
        squeeze_dim_140 = None
        view_default_966 = torch.ops.aten.view.default(squeeze_dim_134, [1, 84, 1, 1])
        squeeze_dim_134 = None
        view_default_970 = torch.ops.aten.view.default(squeeze_dim_128, [1, 84, 1, 1])
        squeeze_dim_128 = None
        view_default_974 = torch.ops.aten.view.default(squeeze_dim_122, [1, 84, 1, 1])
        squeeze_dim_122 = None
        le_scalar_244 = torch.ops.aten.le.Scalar(relu_default_15, 0)
        relu_default_15 = None
        view_default_978 = torch.ops.aten.view.default(squeeze_dim_116, [1, 84, 1, 1])
        squeeze_dim_116 = None
        view_default_982 = torch.ops.aten.view.default(squeeze_dim_110, [1, 84, 1, 1])
        squeeze_dim_110 = None
        view_default_986 = torch.ops.aten.view.default(squeeze_dim_104, [1, 84, 1, 1])
        squeeze_dim_104 = None
        view_default_990 = torch.ops.aten.view.default(squeeze_dim_98, [1, 84, 1, 1])
        squeeze_dim_98 = None
        view_default_994 = torch.ops.aten.view.default(squeeze_dim_92, [1, 84, 1, 1])
        squeeze_dim_92 = None
        view_default_998 = torch.ops.aten.view.default(squeeze_dim_86, [1, 84, 1, 1])
        squeeze_dim_86 = None
        le_scalar_250 = torch.ops.aten.le.Scalar(relu_default_13, 0)
        relu_default_13 = None
        view_default_1002 = torch.ops.aten.view.default(squeeze_dim_80, [1, 84, 1, 1])
        squeeze_dim_80 = None
        view_default_1006 = torch.ops.aten.view.default(squeeze_dim_74, [1, 84, 1, 1])
        squeeze_dim_74 = None
        view_default_1010 = torch.ops.aten.view.default(squeeze_dim_68, [1, 42, 1, 1])
        squeeze_dim_68 = None
        view_default_1014 = torch.ops.aten.view.default(squeeze_dim_62, [1, 42, 1, 1])
        squeeze_dim_62 = None
        view_default_1018 = torch.ops.aten.view.default(squeeze_dim_56, [1, 42, 1, 1])
        squeeze_dim_56 = None
        view_default_1022 = torch.ops.aten.view.default(squeeze_dim_50, [1, 42, 1, 1])
        squeeze_dim_50 = None
        view_default_1026 = torch.ops.aten.view.default(squeeze_dim_44, [1, 42, 1, 1])
        squeeze_dim_44 = None
        view_default_1030 = torch.ops.aten.view.default(squeeze_dim_38, [1, 42, 1, 1])
        squeeze_dim_38 = None
        view_default_1034 = torch.ops.aten.view.default(squeeze_dim_32, [1, 42, 1, 1])
        squeeze_dim_32 = None
        view_default_1038 = torch.ops.aten.view.default(squeeze_dim_26, [1, 42, 1, 1])
        squeeze_dim_26 = None
        view_default_1042 = torch.ops.aten.view.default(squeeze_dim_20, [1, 42, 1, 1])
        squeeze_dim_20 = None
        view_default_1046 = torch.ops.aten.view.default(squeeze_dim_14, [1, 42, 1, 1])
        squeeze_dim_14 = None
        le_scalar_262 = torch.ops.aten.le.Scalar(relu_default_1, 0)
        relu_default_1 = None
        view_default_1050 = torch.ops.aten.view.default(squeeze_dim_8, [1, 42, 1, 1])
        squeeze_dim_8 = None
        view_default_1054 = torch.ops.aten.view.default(squeeze_dim_2, [1, 96, 1, 1])
        squeeze_dim_2 = None
        return [
            add_tensor_1167,
            add_tensor,
            primals_1,
            primals_2,
            primals_3,
            primals_4,
            primals_5,
            primals_6,
            primals_7,
            primals_8,
            primals_9,
            primals_10,
            primals_11,
            primals_12,
            primals_13,
            primals_14,
            primals_15,
            primals_16,
            primals_17,
            primals_18,
            primals_19,
            primals_21,
            primals_22,
            primals_24,
            primals_25,
            primals_26,
            primals_28,
            primals_29,
            primals_31,
            primals_32,
            primals_33,
            primals_35,
            primals_36,
            primals_38,
            primals_39,
            primals_40,
            primals_42,
            primals_43,
            primals_45,
            primals_46,
            primals_47,
            primals_49,
            primals_50,
            primals_51,
            primals_53,
            primals_54,
            primals_55,
            primals_57,
            primals_58,
            primals_60,
            primals_61,
            primals_62,
            primals_64,
            primals_65,
            primals_67,
            primals_68,
            primals_69,
            primals_71,
            primals_72,
            primals_74,
            primals_75,
            primals_76,
            primals_78,
            primals_79,
            primals_81,
            primals_82,
            primals_83,
            primals_85,
            primals_86,
            primals_88,
            primals_89,
            primals_90,
            primals_92,
            primals_93,
            primals_94,
            primals_96,
            primals_97,
            primals_98,
            primals_100,
            primals_101,
            primals_102,
            primals_103,
            primals_104,
            primals_105,
            primals_106,
            primals_107,
            primals_108,
            primals_109,
            primals_111,
            primals_112,
            primals_113,
            primals_115,
            primals_116,
            primals_117,
            primals_119,
            primals_120,
            primals_121,
            primals_123,
            primals_124,
            primals_125,
            primals_127,
            primals_128,
            primals_129,
            primals_131,
            primals_132,
            primals_133,
            primals_135,
            primals_136,
            primals_137,
            primals_139,
            primals_140,
            primals_141,
            primals_143,
            primals_144,
            primals_145,
            primals_147,
            primals_148,
            primals_149,
            primals_150,
            primals_151,
            primals_152,
            primals_153,
            primals_154,
            primals_155,
            primals_157,
            primals_158,
            primals_159,
            primals_161,
            primals_162,
            primals_163,
            primals_165,
            primals_166,
            primals_167,
            primals_169,
            primals_170,
            primals_171,
            primals_173,
            primals_174,
            primals_175,
            primals_177,
            primals_178,
            primals_179,
            primals_181,
            primals_182,
            primals_183,
            primals_185,
            primals_186,
            primals_187,
            primals_189,
            primals_190,
            primals_191,
            primals_193,
            primals_194,
            primals_195,
            primals_196,
            primals_197,
            primals_198,
            primals_199,
            primals_200,
            primals_201,
            primals_203,
            primals_204,
            primals_205,
            primals_207,
            primals_208,
            primals_209,
            primals_211,
            primals_212,
            primals_213,
            primals_215,
            primals_216,
            primals_217,
            primals_219,
            primals_220,
            primals_221,
            primals_223,
            primals_224,
            primals_225,
            primals_227,
            primals_228,
            primals_229,
            primals_231,
            primals_232,
            primals_233,
            primals_235,
            primals_236,
            primals_237,
            primals_239,
            primals_240,
            primals_241,
            primals_242,
            primals_243,
            primals_244,
            primals_245,
            primals_246,
            primals_247,
            primals_249,
            primals_250,
            primals_251,
            primals_253,
            primals_254,
            primals_255,
            primals_257,
            primals_258,
            primals_259,
            primals_261,
            primals_262,
            primals_263,
            primals_265,
            primals_266,
            primals_267,
            primals_269,
            primals_270,
            primals_271,
            primals_273,
            primals_274,
            primals_275,
            primals_277,
            primals_278,
            primals_279,
            primals_281,
            primals_282,
            primals_283,
            primals_285,
            primals_286,
            primals_287,
            primals_288,
            primals_289,
            primals_290,
            primals_291,
            primals_292,
            primals_293,
            primals_295,
            primals_296,
            primals_297,
            primals_299,
            primals_300,
            primals_301,
            primals_303,
            primals_304,
            primals_305,
            primals_307,
            primals_308,
            primals_309,
            primals_311,
            primals_312,
            primals_313,
            primals_315,
            primals_316,
            primals_317,
            primals_319,
            primals_320,
            primals_321,
            primals_323,
            primals_324,
            primals_325,
            primals_327,
            primals_328,
            primals_329,
            primals_331,
            primals_332,
            primals_333,
            primals_334,
            primals_335,
            primals_336,
            primals_337,
            primals_338,
            primals_339,
            primals_341,
            primals_342,
            primals_343,
            primals_345,
            primals_346,
            primals_347,
            primals_349,
            primals_350,
            primals_351,
            primals_353,
            primals_354,
            primals_355,
            primals_357,
            primals_358,
            primals_359,
            primals_361,
            primals_362,
            primals_363,
            primals_365,
            primals_366,
            primals_367,
            primals_369,
            primals_370,
            primals_371,
            primals_373,
            primals_374,
            primals_375,
            primals_377,
            primals_378,
            primals_380,
            primals_381,
            primals_383,
            primals_384,
            primals_386,
            primals_387,
            primals_388,
            primals_390,
            primals_391,
            primals_393,
            primals_394,
            primals_395,
            primals_397,
            primals_398,
            primals_400,
            primals_401,
            primals_402,
            primals_404,
            primals_405,
            primals_407,
            primals_408,
            primals_409,
            primals_411,
            primals_412,
            primals_413,
            primals_415,
            primals_416,
            primals_417,
            primals_419,
            primals_420,
            primals_421,
            primals_422,
            primals_423,
            primals_424,
            primals_425,
            primals_426,
            primals_427,
            primals_428,
            primals_430,
            primals_431,
            primals_432,
            primals_434,
            primals_435,
            primals_436,
            primals_438,
            primals_439,
            primals_440,
            primals_442,
            primals_443,
            primals_444,
            primals_446,
            primals_447,
            primals_448,
            primals_450,
            primals_451,
            primals_452,
            primals_454,
            primals_455,
            primals_456,
            primals_458,
            primals_459,
            primals_460,
            primals_462,
            primals_463,
            primals_464,
            primals_466,
            primals_467,
            primals_468,
            primals_469,
            primals_470,
            primals_471,
            primals_472,
            primals_473,
            primals_474,
            primals_476,
            primals_477,
            primals_478,
            primals_480,
            primals_481,
            primals_482,
            primals_484,
            primals_485,
            primals_486,
            primals_488,
            primals_489,
            primals_490,
            primals_492,
            primals_493,
            primals_494,
            primals_496,
            primals_497,
            primals_498,
            primals_500,
            primals_501,
            primals_502,
            primals_504,
            primals_505,
            primals_506,
            primals_508,
            primals_509,
            primals_510,
            primals_512,
            primals_513,
            primals_514,
            primals_515,
            primals_516,
            primals_517,
            primals_518,
            primals_519,
            primals_520,
            primals_522,
            primals_523,
            primals_524,
            primals_526,
            primals_527,
            primals_528,
            primals_530,
            primals_531,
            primals_532,
            primals_534,
            primals_535,
            primals_536,
            primals_538,
            primals_539,
            primals_540,
            primals_542,
            primals_543,
            primals_544,
            primals_546,
            primals_547,
            primals_548,
            primals_550,
            primals_551,
            primals_552,
            primals_554,
            primals_555,
            primals_556,
            primals_558,
            primals_559,
            primals_560,
            primals_561,
            primals_562,
            primals_563,
            primals_564,
            primals_565,
            primals_566,
            primals_568,
            primals_569,
            primals_570,
            primals_572,
            primals_573,
            primals_574,
            primals_576,
            primals_577,
            primals_578,
            primals_580,
            primals_581,
            primals_582,
            primals_584,
            primals_585,
            primals_586,
            primals_588,
            primals_589,
            primals_590,
            primals_592,
            primals_593,
            primals_594,
            primals_596,
            primals_597,
            primals_598,
            primals_600,
            primals_601,
            primals_602,
            primals_604,
            primals_605,
            primals_606,
            primals_607,
            primals_608,
            primals_609,
            primals_610,
            primals_611,
            primals_612,
            primals_614,
            primals_615,
            primals_616,
            primals_618,
            primals_619,
            primals_620,
            primals_622,
            primals_623,
            primals_624,
            primals_626,
            primals_627,
            primals_628,
            primals_630,
            primals_631,
            primals_632,
            primals_634,
            primals_635,
            primals_636,
            primals_638,
            primals_639,
            primals_640,
            primals_642,
            primals_643,
            primals_644,
            primals_646,
            primals_647,
            primals_648,
            primals_650,
            primals_651,
            primals_652,
            primals_653,
            primals_654,
            primals_655,
            primals_656,
            primals_657,
            primals_658,
            primals_660,
            primals_661,
            primals_662,
            primals_664,
            primals_665,
            primals_666,
            primals_668,
            primals_669,
            primals_670,
            primals_672,
            primals_673,
            primals_674,
            primals_676,
            primals_677,
            primals_678,
            primals_680,
            primals_681,
            primals_682,
            primals_684,
            primals_685,
            primals_686,
            primals_688,
            primals_689,
            primals_690,
            primals_692,
            primals_693,
            primals_694,
            primals_696,
            primals_697,
            primals_699,
            primals_700,
            primals_702,
            primals_703,
            primals_705,
            primals_706,
            primals_707,
            primals_709,
            primals_710,
            primals_712,
            primals_713,
            primals_714,
            primals_716,
            primals_717,
            primals_719,
            primals_720,
            primals_721,
            primals_723,
            primals_724,
            primals_726,
            primals_727,
            primals_728,
            primals_730,
            primals_731,
            primals_732,
            primals_734,
            primals_735,
            primals_736,
            primals_738,
            primals_739,
            primals_740,
            primals_741,
            primals_742,
            primals_743,
            primals_744,
            primals_745,
            primals_746,
            primals_747,
            primals_749,
            primals_750,
            primals_751,
            primals_753,
            primals_754,
            primals_755,
            primals_757,
            primals_758,
            primals_759,
            primals_761,
            primals_762,
            primals_763,
            primals_765,
            primals_766,
            primals_767,
            primals_769,
            primals_770,
            primals_771,
            primals_773,
            primals_774,
            primals_775,
            primals_777,
            primals_778,
            primals_779,
            primals_781,
            primals_782,
            primals_783,
            primals_785,
            primals_786,
            primals_787,
            primals_788,
            primals_789,
            primals_790,
            primals_791,
            primals_792,
            primals_793,
            primals_795,
            primals_796,
            primals_797,
            primals_799,
            primals_800,
            primals_801,
            primals_803,
            primals_804,
            primals_805,
            primals_807,
            primals_808,
            primals_809,
            primals_811,
            primals_812,
            primals_813,
            primals_815,
            primals_816,
            primals_817,
            primals_819,
            primals_820,
            primals_821,
            primals_823,
            primals_824,
            primals_825,
            primals_827,
            primals_828,
            primals_829,
            primals_831,
            primals_832,
            primals_833,
            primals_834,
            primals_835,
            primals_836,
            primals_837,
            primals_838,
            primals_839,
            primals_841,
            primals_842,
            primals_843,
            primals_845,
            primals_846,
            primals_847,
            primals_849,
            primals_850,
            primals_851,
            primals_853,
            primals_854,
            primals_855,
            primals_857,
            primals_858,
            primals_859,
            primals_861,
            primals_862,
            primals_863,
            primals_865,
            primals_866,
            primals_867,
            primals_869,
            primals_870,
            primals_871,
            primals_873,
            primals_874,
            primals_875,
            primals_877,
            primals_878,
            primals_879,
            primals_880,
            primals_881,
            primals_882,
            primals_883,
            primals_884,
            primals_885,
            primals_887,
            primals_888,
            primals_889,
            primals_891,
            primals_892,
            primals_893,
            primals_895,
            primals_896,
            primals_897,
            primals_899,
            primals_900,
            primals_901,
            primals_903,
            primals_904,
            primals_905,
            primals_907,
            primals_908,
            primals_909,
            primals_911,
            primals_912,
            primals_913,
            primals_915,
            primals_916,
            primals_917,
            primals_919,
            primals_920,
            primals_921,
            primals_923,
            primals_924,
            primals_925,
            primals_926,
            primals_927,
            primals_928,
            primals_929,
            primals_930,
            primals_931,
            primals_933,
            primals_934,
            primals_935,
            primals_937,
            primals_938,
            primals_939,
            primals_941,
            primals_942,
            primals_943,
            primals_945,
            primals_946,
            primals_947,
            primals_949,
            primals_950,
            primals_951,
            primals_953,
            primals_954,
            primals_955,
            primals_957,
            primals_958,
            primals_959,
            primals_961,
            primals_962,
            primals_963,
            primals_965,
            primals_966,
            primals_967,
            primals_969,
            primals_970,
            primals_971,
            primals_972,
            primals_973,
            primals_974,
            primals_975,
            primals_976,
            primals_977,
            primals_979,
            primals_980,
            primals_981,
            primals_983,
            primals_984,
            primals_985,
            primals_987,
            primals_988,
            primals_989,
            primals_991,
            primals_992,
            primals_993,
            primals_995,
            primals_996,
            primals_997,
            primals_999,
            primals_1000,
            primals_1001,
            primals_1003,
            primals_1004,
            primals_1005,
            primals_1007,
            primals_1008,
            primals_1009,
            primals_1011,
            primals_1012,
            primals_1013,
            primals_1543,
            primals_1547,
            convolution_default,
            squeeze_dim_5,
            relu_default,
            convolution_default_1,
            squeeze_dim_11,
            constant_pad_nd_default,
            convolution_default_2,
            convolution_default_3,
            squeeze_dim_17,
            relu_default_2,
            convolution_default_4,
            convolution_default_5,
            squeeze_dim_23,
            constant_pad_nd_default_1,
            convolution_default_6,
            convolution_default_7,
            squeeze_dim_29,
            relu_default_4,
            convolution_default_8,
            convolution_default_9,
            squeeze_dim_35,
            add_tensor_25,
            constant_pad_nd_default_2,
            getitem_1,
            convolution_default_10,
            convolution_default_11,
            squeeze_dim_41,
            relu_default_6,
            convolution_default_12,
            convolution_default_13,
            squeeze_dim_47,
            constant_pad_nd_default_4,
            constant_pad_nd_default_5,
            convolution_default_14,
            convolution_default_15,
            squeeze_dim_53,
            relu_default_8,
            convolution_default_16,
            convolution_default_17,
            squeeze_dim_59,
            convolution_default_18,
            convolution_default_19,
            squeeze_dim_65,
            relu_default_10,
            convolution_default_20,
            convolution_default_21,
            squeeze_dim_71,
            relu_default_11,
            convolution_default_22,
            squeeze_dim_77,
            avg_pool2d_default_2,
            constant_pad_nd_default_7,
            avg_pool2d_default_3,
            cat_default_1,
            squeeze_dim_83,
            constant_pad_nd_default_8,
            convolution_default_25,
            convolution_default_26,
            squeeze_dim_89,
            relu_default_14,
            convolution_default_27,
            convolution_default_28,
            squeeze_dim_95,
            constant_pad_nd_default_9,
            convolution_default_29,
            convolution_default_30,
            squeeze_dim_101,
            relu_default_16,
            convolution_default_31,
            convolution_default_32,
            squeeze_dim_107,
            add_tensor_78,
            constant_pad_nd_default_10,
            getitem_5,
            convolution_default_33,
            convolution_default_34,
            squeeze_dim_113,
            relu_default_18,
            convolution_default_35,
            convolution_default_36,
            squeeze_dim_119,
            constant_pad_nd_default_12,
            constant_pad_nd_default_13,
            convolution_default_37,
            convolution_default_38,
            squeeze_dim_125,
            relu_default_20,
            convolution_default_39,
            convolution_default_40,
            squeeze_dim_131,
            convolution_default_41,
            convolution_default_42,
            squeeze_dim_137,
            relu_default_22,
            convolution_default_43,
            convolution_default_44,
            squeeze_dim_143,
            avg_pool2d_default_6,
            constant_pad_nd_default_15,
            avg_pool2d_default_7,
            cat_default_3,
            mean_dim_24,
            reciprocal_default_24,
            relu_default_24,
            convolution_default_47,
            mean_dim_25,
            reciprocal_default_25,
            convolution_default_48,
            convolution_default_49,
            squeeze_dim_161,
            relu_default_26,
            convolution_default_50,
            convolution_default_51,
            squeeze_dim_167,
            convolution_default_52,
            convolution_default_53,
            squeeze_dim_173,
            relu_default_28,
            convolution_default_54,
            convolution_default_55,
            squeeze_dim_179,
            convolution_default_56,
            convolution_default_57,
            squeeze_dim_185,
            relu_default_30,
            convolution_default_58,
            convolution_default_59,
            squeeze_dim_191,
            convolution_default_60,
            convolution_default_61,
            squeeze_dim_197,
            relu_default_32,
            convolution_default_62,
            convolution_default_63,
            squeeze_dim_203,
            convolution_default_64,
            convolution_default_65,
            squeeze_dim_209,
            relu_default_34,
            convolution_default_66,
            convolution_default_67,
            squeeze_dim_215,
            convolution_default_68,
            mean_dim_36,
            reciprocal_default_36,
            relu_default_36,
            convolution_default_69,
            mean_dim_37,
            reciprocal_default_37,
            convolution_default_70,
            convolution_default_71,
            squeeze_dim_233,
            relu_default_38,
            convolution_default_72,
            convolution_default_73,
            squeeze_dim_239,
            convolution_default_74,
            convolution_default_75,
            squeeze_dim_245,
            relu_default_40,
            convolution_default_76,
            convolution_default_77,
            squeeze_dim_251,
            convolution_default_78,
            convolution_default_79,
            squeeze_dim_257,
            relu_default_42,
            convolution_default_80,
            convolution_default_81,
            squeeze_dim_263,
            convolution_default_82,
            convolution_default_83,
            squeeze_dim_269,
            relu_default_44,
            convolution_default_84,
            convolution_default_85,
            squeeze_dim_275,
            convolution_default_86,
            convolution_default_87,
            squeeze_dim_281,
            relu_default_46,
            convolution_default_88,
            convolution_default_89,
            squeeze_dim_287,
            convolution_default_90,
            mean_dim_48,
            reciprocal_default_48,
            relu_default_48,
            convolution_default_91,
            mean_dim_49,
            reciprocal_default_49,
            convolution_default_92,
            convolution_default_93,
            squeeze_dim_305,
            relu_default_50,
            convolution_default_94,
            convolution_default_95,
            squeeze_dim_311,
            convolution_default_96,
            convolution_default_97,
            squeeze_dim_317,
            relu_default_52,
            convolution_default_98,
            convolution_default_99,
            squeeze_dim_323,
            convolution_default_100,
            convolution_default_101,
            squeeze_dim_329,
            relu_default_54,
            convolution_default_102,
            convolution_default_103,
            squeeze_dim_335,
            convolution_default_104,
            convolution_default_105,
            squeeze_dim_341,
            relu_default_56,
            convolution_default_106,
            convolution_default_107,
            squeeze_dim_347,
            convolution_default_108,
            convolution_default_109,
            squeeze_dim_353,
            relu_default_58,
            convolution_default_110,
            convolution_default_111,
            squeeze_dim_359,
            convolution_default_112,
            mean_dim_60,
            reciprocal_default_60,
            relu_default_60,
            convolution_default_113,
            mean_dim_61,
            reciprocal_default_61,
            convolution_default_114,
            convolution_default_115,
            squeeze_dim_377,
            relu_default_62,
            convolution_default_116,
            convolution_default_117,
            squeeze_dim_383,
            convolution_default_118,
            convolution_default_119,
            squeeze_dim_389,
            relu_default_64,
            convolution_default_120,
            convolution_default_121,
            squeeze_dim_395,
            convolution_default_122,
            convolution_default_123,
            squeeze_dim_401,
            relu_default_66,
            convolution_default_124,
            convolution_default_125,
            squeeze_dim_407,
            convolution_default_126,
            convolution_default_127,
            squeeze_dim_413,
            relu_default_68,
            convolution_default_128,
            convolution_default_129,
            squeeze_dim_419,
            convolution_default_130,
            convolution_default_131,
            squeeze_dim_425,
            relu_default_70,
            convolution_default_132,
            convolution_default_133,
            squeeze_dim_431,
            convolution_default_134,
            mean_dim_72,
            reciprocal_default_72,
            relu_default_72,
            convolution_default_135,
            mean_dim_73,
            reciprocal_default_73,
            convolution_default_136,
            convolution_default_137,
            squeeze_dim_449,
            relu_default_74,
            convolution_default_138,
            convolution_default_139,
            squeeze_dim_455,
            convolution_default_140,
            convolution_default_141,
            squeeze_dim_461,
            relu_default_76,
            convolution_default_142,
            convolution_default_143,
            squeeze_dim_467,
            convolution_default_144,
            convolution_default_145,
            squeeze_dim_473,
            relu_default_78,
            convolution_default_146,
            convolution_default_147,
            squeeze_dim_479,
            convolution_default_148,
            convolution_default_149,
            squeeze_dim_485,
            relu_default_80,
            convolution_default_150,
            convolution_default_151,
            squeeze_dim_491,
            convolution_default_152,
            convolution_default_153,
            squeeze_dim_497,
            relu_default_82,
            convolution_default_154,
            convolution_default_155,
            squeeze_dim_503,
            convolution_default_156,
            mean_dim_84,
            reciprocal_default_84,
            relu_default_84,
            convolution_default_157,
            mean_dim_85,
            reciprocal_default_85,
            convolution_default_158,
            convolution_default_159,
            squeeze_dim_521,
            relu_default_86,
            convolution_default_160,
            convolution_default_161,
            squeeze_dim_527,
            convolution_default_162,
            convolution_default_163,
            squeeze_dim_533,
            relu_default_88,
            convolution_default_164,
            convolution_default_165,
            squeeze_dim_539,
            convolution_default_166,
            convolution_default_167,
            squeeze_dim_545,
            relu_default_90,
            convolution_default_168,
            convolution_default_169,
            squeeze_dim_551,
            convolution_default_170,
            convolution_default_171,
            squeeze_dim_557,
            relu_default_92,
            convolution_default_172,
            convolution_default_173,
            squeeze_dim_563,
            convolution_default_174,
            convolution_default_175,
            squeeze_dim_569,
            relu_default_94,
            convolution_default_176,
            convolution_default_177,
            squeeze_dim_575,
            convolution_default_178,
            squeeze_dim_581,
            relu_default_96,
            convolution_default_179,
            squeeze_dim_587,
            constant_pad_nd_default_16,
            convolution_default_180,
            convolution_default_181,
            squeeze_dim_593,
            relu_default_98,
            convolution_default_182,
            convolution_default_183,
            squeeze_dim_599,
            constant_pad_nd_default_17,
            convolution_default_184,
            convolution_default_185,
            squeeze_dim_605,
            relu_default_100,
            convolution_default_186,
            convolution_default_187,
            squeeze_dim_611,
            add_tensor_449,
            constant_pad_nd_default_18,
            getitem_9,
            convolution_default_188,
            convolution_default_189,
            squeeze_dim_617,
            relu_default_102,
            convolution_default_190,
            convolution_default_191,
            squeeze_dim_623,
            constant_pad_nd_default_20,
            constant_pad_nd_default_21,
            convolution_default_192,
            convolution_default_193,
            squeeze_dim_629,
            relu_default_104,
            convolution_default_194,
            convolution_default_195,
            squeeze_dim_635,
            convolution_default_196,
            convolution_default_197,
            squeeze_dim_641,
            relu_default_106,
            convolution_default_198,
            convolution_default_199,
            squeeze_dim_647,
            avg_pool2d_default_28,
            constant_pad_nd_default_23,
            avg_pool2d_default_29,
            cat_default_11,
            mean_dim_108,
            reciprocal_default_108,
            relu_default_108,
            convolution_default_202,
            mean_dim_109,
            reciprocal_default_109,
            convolution_default_203,
            convolution_default_204,
            squeeze_dim_665,
            relu_default_110,
            convolution_default_205,
            convolution_default_206,
            squeeze_dim_671,
            convolution_default_207,
            convolution_default_208,
            squeeze_dim_677,
            relu_default_112,
            convolution_default_209,
            convolution_default_210,
            squeeze_dim_683,
            convolution_default_211,
            convolution_default_212,
            squeeze_dim_689,
            relu_default_114,
            convolution_default_213,
            convolution_default_214,
            squeeze_dim_695,
            convolution_default_215,
            convolution_default_216,
            squeeze_dim_701,
            relu_default_116,
            convolution_default_217,
            convolution_default_218,
            squeeze_dim_707,
            convolution_default_219,
            convolution_default_220,
            squeeze_dim_713,
            relu_default_118,
            convolution_default_221,
            convolution_default_222,
            squeeze_dim_719,
            convolution_default_223,
            mean_dim_120,
            reciprocal_default_120,
            relu_default_120,
            convolution_default_224,
            mean_dim_121,
            reciprocal_default_121,
            convolution_default_225,
            convolution_default_226,
            squeeze_dim_737,
            relu_default_122,
            convolution_default_227,
            convolution_default_228,
            squeeze_dim_743,
            convolution_default_229,
            convolution_default_230,
            squeeze_dim_749,
            relu_default_124,
            convolution_default_231,
            convolution_default_232,
            squeeze_dim_755,
            convolution_default_233,
            convolution_default_234,
            squeeze_dim_761,
            relu_default_126,
            convolution_default_235,
            convolution_default_236,
            squeeze_dim_767,
            convolution_default_237,
            convolution_default_238,
            squeeze_dim_773,
            relu_default_128,
            convolution_default_239,
            convolution_default_240,
            squeeze_dim_779,
            convolution_default_241,
            convolution_default_242,
            squeeze_dim_785,
            relu_default_130,
            convolution_default_243,
            convolution_default_244,
            squeeze_dim_791,
            convolution_default_245,
            mean_dim_132,
            reciprocal_default_132,
            relu_default_132,
            convolution_default_246,
            mean_dim_133,
            reciprocal_default_133,
            convolution_default_247,
            convolution_default_248,
            squeeze_dim_809,
            relu_default_134,
            convolution_default_249,
            convolution_default_250,
            squeeze_dim_815,
            convolution_default_251,
            convolution_default_252,
            squeeze_dim_821,
            relu_default_136,
            convolution_default_253,
            convolution_default_254,
            squeeze_dim_827,
            convolution_default_255,
            convolution_default_256,
            squeeze_dim_833,
            relu_default_138,
            convolution_default_257,
            convolution_default_258,
            squeeze_dim_839,
            convolution_default_259,
            convolution_default_260,
            squeeze_dim_845,
            relu_default_140,
            convolution_default_261,
            convolution_default_262,
            squeeze_dim_851,
            convolution_default_263,
            convolution_default_264,
            squeeze_dim_857,
            relu_default_142,
            convolution_default_265,
            convolution_default_266,
            squeeze_dim_863,
            convolution_default_267,
            mean_dim_144,
            reciprocal_default_144,
            relu_default_144,
            convolution_default_268,
            mean_dim_145,
            reciprocal_default_145,
            convolution_default_269,
            convolution_default_270,
            squeeze_dim_881,
            relu_default_146,
            convolution_default_271,
            convolution_default_272,
            squeeze_dim_887,
            convolution_default_273,
            convolution_default_274,
            squeeze_dim_893,
            relu_default_148,
            convolution_default_275,
            convolution_default_276,
            squeeze_dim_899,
            convolution_default_277,
            convolution_default_278,
            squeeze_dim_905,
            relu_default_150,
            convolution_default_279,
            convolution_default_280,
            squeeze_dim_911,
            convolution_default_281,
            convolution_default_282,
            squeeze_dim_917,
            relu_default_152,
            convolution_default_283,
            convolution_default_284,
            squeeze_dim_923,
            convolution_default_285,
            convolution_default_286,
            squeeze_dim_929,
            relu_default_154,
            convolution_default_287,
            convolution_default_288,
            squeeze_dim_935,
            convolution_default_289,
            mean_dim_156,
            reciprocal_default_156,
            relu_default_156,
            convolution_default_290,
            mean_dim_157,
            reciprocal_default_157,
            convolution_default_291,
            convolution_default_292,
            squeeze_dim_953,
            relu_default_158,
            convolution_default_293,
            convolution_default_294,
            squeeze_dim_959,
            convolution_default_295,
            convolution_default_296,
            squeeze_dim_965,
            relu_default_160,
            convolution_default_297,
            convolution_default_298,
            squeeze_dim_971,
            convolution_default_299,
            convolution_default_300,
            squeeze_dim_977,
            relu_default_162,
            convolution_default_301,
            convolution_default_302,
            squeeze_dim_983,
            convolution_default_303,
            convolution_default_304,
            squeeze_dim_989,
            relu_default_164,
            convolution_default_305,
            convolution_default_306,
            squeeze_dim_995,
            convolution_default_307,
            convolution_default_308,
            squeeze_dim_1001,
            relu_default_166,
            convolution_default_309,
            convolution_default_310,
            squeeze_dim_1007,
            convolution_default_311,
            mean_dim_168,
            reciprocal_default_168,
            relu_default_168,
            convolution_default_312,
            mean_dim_169,
            reciprocal_default_169,
            convolution_default_313,
            convolution_default_314,
            squeeze_dim_1025,
            relu_default_170,
            convolution_default_315,
            convolution_default_316,
            squeeze_dim_1031,
            convolution_default_317,
            convolution_default_318,
            squeeze_dim_1037,
            relu_default_172,
            convolution_default_319,
            convolution_default_320,
            squeeze_dim_1043,
            convolution_default_321,
            convolution_default_322,
            squeeze_dim_1049,
            relu_default_174,
            convolution_default_323,
            convolution_default_324,
            squeeze_dim_1055,
            convolution_default_325,
            convolution_default_326,
            squeeze_dim_1061,
            relu_default_176,
            convolution_default_327,
            convolution_default_328,
            squeeze_dim_1067,
            convolution_default_329,
            convolution_default_330,
            squeeze_dim_1073,
            relu_default_178,
            convolution_default_331,
            convolution_default_332,
            squeeze_dim_1079,
            convolution_default_333,
            squeeze_dim_1085,
            relu_default_180,
            convolution_default_334,
            squeeze_dim_1091,
            constant_pad_nd_default_24,
            convolution_default_335,
            convolution_default_336,
            squeeze_dim_1097,
            relu_default_182,
            convolution_default_337,
            convolution_default_338,
            squeeze_dim_1103,
            constant_pad_nd_default_25,
            convolution_default_339,
            convolution_default_340,
            squeeze_dim_1109,
            relu_default_184,
            convolution_default_341,
            convolution_default_342,
            squeeze_dim_1115,
            add_tensor_820,
            constant_pad_nd_default_26,
            getitem_13,
            convolution_default_343,
            convolution_default_344,
            squeeze_dim_1121,
            relu_default_186,
            convolution_default_345,
            convolution_default_346,
            squeeze_dim_1127,
            constant_pad_nd_default_28,
            constant_pad_nd_default_29,
            convolution_default_347,
            convolution_default_348,
            squeeze_dim_1133,
            relu_default_188,
            convolution_default_349,
            convolution_default_350,
            squeeze_dim_1139,
            convolution_default_351,
            convolution_default_352,
            squeeze_dim_1145,
            relu_default_190,
            convolution_default_353,
            convolution_default_354,
            squeeze_dim_1151,
            avg_pool2d_default_50,
            constant_pad_nd_default_31,
            avg_pool2d_default_51,
            cat_default_19,
            mean_dim_192,
            reciprocal_default_192,
            relu_default_192,
            convolution_default_357,
            mean_dim_193,
            reciprocal_default_193,
            convolution_default_358,
            convolution_default_359,
            squeeze_dim_1169,
            relu_default_194,
            convolution_default_360,
            convolution_default_361,
            squeeze_dim_1175,
            convolution_default_362,
            convolution_default_363,
            squeeze_dim_1181,
            relu_default_196,
            convolution_default_364,
            convolution_default_365,
            squeeze_dim_1187,
            convolution_default_366,
            convolution_default_367,
            squeeze_dim_1193,
            relu_default_198,
            convolution_default_368,
            convolution_default_369,
            squeeze_dim_1199,
            convolution_default_370,
            convolution_default_371,
            squeeze_dim_1205,
            relu_default_200,
            convolution_default_372,
            convolution_default_373,
            squeeze_dim_1211,
            convolution_default_374,
            convolution_default_375,
            squeeze_dim_1217,
            relu_default_202,
            convolution_default_376,
            convolution_default_377,
            squeeze_dim_1223,
            convolution_default_378,
            mean_dim_204,
            reciprocal_default_204,
            relu_default_204,
            convolution_default_379,
            mean_dim_205,
            reciprocal_default_205,
            convolution_default_380,
            convolution_default_381,
            squeeze_dim_1241,
            relu_default_206,
            convolution_default_382,
            convolution_default_383,
            squeeze_dim_1247,
            convolution_default_384,
            convolution_default_385,
            squeeze_dim_1253,
            relu_default_208,
            convolution_default_386,
            convolution_default_387,
            squeeze_dim_1259,
            convolution_default_388,
            convolution_default_389,
            squeeze_dim_1265,
            relu_default_210,
            convolution_default_390,
            convolution_default_391,
            squeeze_dim_1271,
            convolution_default_392,
            convolution_default_393,
            squeeze_dim_1277,
            relu_default_212,
            convolution_default_394,
            convolution_default_395,
            squeeze_dim_1283,
            convolution_default_396,
            convolution_default_397,
            squeeze_dim_1289,
            relu_default_214,
            convolution_default_398,
            convolution_default_399,
            squeeze_dim_1295,
            convolution_default_400,
            mean_dim_216,
            reciprocal_default_216,
            relu_default_216,
            convolution_default_401,
            mean_dim_217,
            reciprocal_default_217,
            convolution_default_402,
            convolution_default_403,
            squeeze_dim_1313,
            relu_default_218,
            convolution_default_404,
            convolution_default_405,
            squeeze_dim_1319,
            convolution_default_406,
            convolution_default_407,
            squeeze_dim_1325,
            relu_default_220,
            convolution_default_408,
            convolution_default_409,
            squeeze_dim_1331,
            convolution_default_410,
            convolution_default_411,
            squeeze_dim_1337,
            relu_default_222,
            convolution_default_412,
            convolution_default_413,
            squeeze_dim_1343,
            convolution_default_414,
            convolution_default_415,
            squeeze_dim_1349,
            relu_default_224,
            convolution_default_416,
            convolution_default_417,
            squeeze_dim_1355,
            convolution_default_418,
            convolution_default_419,
            squeeze_dim_1361,
            relu_default_226,
            convolution_default_420,
            convolution_default_421,
            squeeze_dim_1367,
            convolution_default_422,
            mean_dim_228,
            reciprocal_default_228,
            relu_default_228,
            convolution_default_423,
            mean_dim_229,
            reciprocal_default_229,
            convolution_default_424,
            convolution_default_425,
            squeeze_dim_1385,
            relu_default_230,
            convolution_default_426,
            convolution_default_427,
            squeeze_dim_1391,
            convolution_default_428,
            convolution_default_429,
            squeeze_dim_1397,
            relu_default_232,
            convolution_default_430,
            convolution_default_431,
            squeeze_dim_1403,
            convolution_default_432,
            convolution_default_433,
            squeeze_dim_1409,
            relu_default_234,
            convolution_default_434,
            convolution_default_435,
            squeeze_dim_1415,
            convolution_default_436,
            convolution_default_437,
            squeeze_dim_1421,
            relu_default_236,
            convolution_default_438,
            convolution_default_439,
            squeeze_dim_1427,
            convolution_default_440,
            convolution_default_441,
            squeeze_dim_1433,
            relu_default_238,
            convolution_default_442,
            convolution_default_443,
            squeeze_dim_1439,
            convolution_default_444,
            mean_dim_240,
            reciprocal_default_240,
            relu_default_240,
            convolution_default_445,
            mean_dim_241,
            reciprocal_default_241,
            convolution_default_446,
            convolution_default_447,
            squeeze_dim_1457,
            relu_default_242,
            convolution_default_448,
            convolution_default_449,
            squeeze_dim_1463,
            convolution_default_450,
            convolution_default_451,
            squeeze_dim_1469,
            relu_default_244,
            convolution_default_452,
            convolution_default_453,
            squeeze_dim_1475,
            convolution_default_454,
            convolution_default_455,
            squeeze_dim_1481,
            relu_default_246,
            convolution_default_456,
            convolution_default_457,
            squeeze_dim_1487,
            convolution_default_458,
            convolution_default_459,
            squeeze_dim_1493,
            relu_default_248,
            convolution_default_460,
            convolution_default_461,
            squeeze_dim_1499,
            convolution_default_462,
            convolution_default_463,
            squeeze_dim_1505,
            relu_default_250,
            convolution_default_464,
            convolution_default_465,
            squeeze_dim_1511,
            convolution_default_466,
            mean_dim_252,
            reciprocal_default_252,
            relu_default_252,
            convolution_default_467,
            mean_dim_253,
            reciprocal_default_253,
            convolution_default_468,
            convolution_default_469,
            squeeze_dim_1529,
            relu_default_254,
            convolution_default_470,
            convolution_default_471,
            squeeze_dim_1535,
            convolution_default_472,
            convolution_default_473,
            squeeze_dim_1541,
            relu_default_256,
            convolution_default_474,
            convolution_default_475,
            squeeze_dim_1547,
            convolution_default_476,
            convolution_default_477,
            squeeze_dim_1553,
            relu_default_258,
            convolution_default_478,
            convolution_default_479,
            squeeze_dim_1559,
            convolution_default_480,
            convolution_default_481,
            squeeze_dim_1565,
            relu_default_260,
            convolution_default_482,
            convolution_default_483,
            squeeze_dim_1571,
            convolution_default_484,
            convolution_default_485,
            squeeze_dim_1577,
            relu_default_262,
            convolution_default_486,
            convolution_default_487,
            squeeze_dim_1583,
            view_default,
            permute_default_1,
            le_scalar,
            view_default_2,
            view_default_6,
            view_default_10,
            view_default_14,
            view_default_18,
            view_default_22,
            view_default_26,
            view_default_30,
            view_default_34,
            view_default_38,
            view_default_50,
            view_default_54,
            view_default_58,
            view_default_62,
            view_default_66,
            view_default_70,
            view_default_74,
            view_default_78,
            view_default_82,
            view_default_86,
            view_default_98,
            view_default_102,
            view_default_106,
            view_default_110,
            view_default_114,
            view_default_118,
            view_default_122,
            view_default_126,
            view_default_130,
            view_default_134,
            view_default_146,
            view_default_150,
            view_default_154,
            view_default_158,
            view_default_162,
            view_default_166,
            view_default_170,
            view_default_174,
            view_default_178,
            view_default_182,
            view_default_194,
            view_default_198,
            view_default_202,
            view_default_206,
            view_default_210,
            view_default_214,
            view_default_218,
            view_default_222,
            view_default_226,
            view_default_230,
            view_default_242,
            view_default_246,
            view_default_250,
            view_default_254,
            view_default_258,
            view_default_262,
            view_default_266,
            view_default_270,
            view_default_274,
            view_default_278,
            view_default_290,
            view_default_294,
            view_default_298,
            view_default_302,
            le_scalar_76,
            view_default_306,
            view_default_310,
            view_default_314,
            view_default_318,
            view_default_322,
            view_default_326,
            le_scalar_82,
            view_default_330,
            view_default_334,
            view_default_338,
            view_default_342,
            view_default_346,
            view_default_350,
            view_default_354,
            view_default_358,
            view_default_362,
            view_default_366,
            view_default_370,
            view_default_374,
            view_default_386,
            view_default_390,
            view_default_394,
            view_default_398,
            view_default_402,
            view_default_406,
            view_default_410,
            view_default_414,
            view_default_418,
            view_default_422,
            view_default_434,
            view_default_438,
            view_default_442,
            view_default_446,
            view_default_450,
            view_default_454,
            view_default_458,
            view_default_462,
            view_default_466,
            view_default_470,
            view_default_482,
            view_default_486,
            view_default_490,
            view_default_494,
            view_default_498,
            view_default_502,
            view_default_506,
            view_default_510,
            view_default_514,
            view_default_518,
            view_default_530,
            view_default_534,
            view_default_538,
            view_default_542,
            view_default_546,
            view_default_550,
            view_default_554,
            view_default_558,
            view_default_562,
            view_default_566,
            view_default_578,
            view_default_582,
            view_default_586,
            view_default_590,
            view_default_594,
            view_default_598,
            view_default_602,
            view_default_606,
            view_default_610,
            view_default_614,
            view_default_626,
            view_default_630,
            view_default_634,
            view_default_638,
            le_scalar_160,
            view_default_642,
            view_default_646,
            view_default_650,
            view_default_654,
            view_default_658,
            view_default_662,
            le_scalar_166,
            view_default_666,
            view_default_670,
            view_default_674,
            view_default_678,
            view_default_682,
            view_default_686,
            view_default_690,
            view_default_694,
            view_default_698,
            view_default_702,
            view_default_706,
            view_default_710,
            view_default_722,
            view_default_726,
            view_default_730,
            view_default_734,
            view_default_738,
            view_default_742,
            view_default_746,
            view_default_750,
            view_default_754,
            view_default_758,
            view_default_770,
            view_default_774,
            view_default_778,
            view_default_782,
            view_default_786,
            view_default_790,
            view_default_794,
            view_default_798,
            view_default_802,
            view_default_806,
            view_default_818,
            view_default_822,
            view_default_826,
            view_default_830,
            view_default_834,
            view_default_838,
            view_default_842,
            view_default_846,
            view_default_850,
            view_default_854,
            view_default_866,
            view_default_870,
            view_default_874,
            view_default_878,
            view_default_882,
            view_default_886,
            view_default_890,
            view_default_894,
            view_default_898,
            view_default_902,
            view_default_914,
            view_default_918,
            view_default_922,
            view_default_926,
            view_default_930,
            view_default_934,
            view_default_938,
            view_default_942,
            view_default_946,
            view_default_950,
            view_default_962,
            view_default_966,
            view_default_970,
            view_default_974,
            le_scalar_244,
            view_default_978,
            view_default_982,
            view_default_986,
            view_default_990,
            view_default_994,
            view_default_998,
            le_scalar_250,
            view_default_1002,
            view_default_1006,
            view_default_1010,
            view_default_1014,
            view_default_1018,
            view_default_1022,
            view_default_1026,
            view_default_1030,
            view_default_1034,
            view_default_1038,
            view_default_1042,
            view_default_1046,
            le_scalar_262,
            view_default_1050,
            view_default_1054,
        ]


args = [
    ((42, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((96, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((96, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((96, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((84, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((84, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((84, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((84, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((336, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((672, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((96, 3, 3, 3), (27, 9, 3, 1), torch.float32, "cuda"),
    ((42, 96, 1, 1), (96, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 42, 1, 1), (42, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((42, 42, 1, 1), (42, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 96, 1, 1), (96, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((42, 42, 1, 1), (42, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 96, 1, 1), (96, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((42, 42, 1, 1), (42, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 96, 1, 1), (96, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((42, 42, 1, 1), (42, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((42, 42, 1, 1), (42, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((42, 42, 1, 1), (42, 1, 1, 1), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((84, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((42, 96, 1, 1), (96, 1, 1, 1), torch.float32, "cuda"),
    ((42, 96, 1, 1), (96, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((84, 84, 1, 1), (84, 1, 1, 1), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((84, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((336, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1344, 1, 1), (1344, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1344, 1, 1), (1344, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((672, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 7, 7), (49, 49, 7, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 2688, 1, 1), (2688, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 2688, 1, 1), (2688, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 4032, 1, 1), (4032, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, "cuda"),
    ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((1000, 4032), (4032, 1), torch.float32, "cuda"),
    ((1000,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((42,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((84,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((168,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((336,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((672,), (1,), torch.float32, "cuda"),
    ((16, 3, 331, 331), (328683, 109561, 331, 1), torch.float32, "cuda"),
    ((), (), torch.int64, "cuda"),
    ((96,), (1,), torch.float32, "cuda"),
    ((96,), (1,), torch.float32, "cuda"),
    ((96,), (1,), torch.float32, "cuda"),
    ((96,), (1,), torch.float32, "cuda"),
]
args = [
    rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args
]
mod = make_fx(Repro())(*args)


from functools import partial

from functorch.compile import minifier

from torchinductor.debug_utils import dump_state_inductor
from torchinductor.debug_utils import inductor_fails
from torchinductor.debug_utils import isolate_inductor_fails

env_variables = {"CUDA_VISIBLE_DEVICES": "1"}

minifier(
    mod,
    args,
    module_fails=partial(isolate_inductor_fails, env=env_variables),
    dump_state=dump_state_inductor,
)
