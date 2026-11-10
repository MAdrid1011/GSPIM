"""Workload paths, sources, revisions and parameters shared by preparation and runs."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache/gspim"
SCENES = (
    "coffee_martini", "cook_spinach", "sear_steak", "flame_steak",
    "dance3", "actor1_4",
)
SCENE_FRAMES = {
    "coffee_martini": 300, "cook_spinach": 300, "flame_steak": 300,
    "sear_steak": 300, "dance3": 300, "actor1_4": 200,
}
FRAMES = 300


@dataclass(frozen=True)
class Upstream:
    name: str
    url: str
    commit: str


EX4DGS = Upstream("Ex4DGS", "https://github.com/juno181/Ex4DGS.git",
                  "7fac64997164fde64732c80261e56adf66ca14df")
ORIGINAL4DGS = Upstream("4d-gaussian-splatting",
    "https://github.com/LogosIntelligenceLab/4d-gaussian-splatting.git",
    "121e360ce23d6e9ca7734072393414c844dfaea2")
FUSED_SSIM = Upstream("fused-ssim", "https://github.com/rahul-goel/fused-ssim.git",
                     "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38")
RAMULATOR = Upstream("ramulator2", "https://github.com/CMU-SAFARI/ramulator2.git",
                    "1bb52ceed30cc607a2d328589815eaea976c0224")
GOLDEN_RATIO = Upstream("Golden-Ratio", "https://github.com/MAdrid1011/Golden-Ratio.git",
                       "437fd7008436e912a035bc6484e2d181c88d840b")
UPSTREAMS = (RAMULATOR, GOLDEN_RATIO)
DATA_URL = "https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/{scene}.zip"

# Official Mobile-Stage and ENeRF-Outdoor camera/video packages use local inputs.
# Original archive digests identify the supplied camera/video packages.
OFFICIAL_SCENE_SOURCES = {
    "dance3": {
        "name": "Mobile-Stage",
        "url": "https://docs.google.com/forms/d/e/1FAIpQLSeEbjuTV7w0lfryl-9FPX1VteuPGkbqjDvxXebY02Tm6BMejQ/viewform?usp=sf_link",
        "archive_sha256": "225886d2b56db95b94496081ce82fd096ccf06a4559469f91858c9838548904a",
        "scene_archive_sha256": "22e86a75852b47cca36bd7405675905088f8064a19220c9eddeede7dfc0641ad",
    },
    "actor1_4": {
        "name": "ENeRF-Outdoor",
        "url": "https://docs.google.com/forms/d/e/1FAIpQLSeFAgcnJEDYEaco4vA2QsD-bM8V8CRFuiR0QWbi_3uXmXtUKg/viewform?usp=sf_link",
        "archive_sha256": "409a335cea24c138c8a9c3fdd96b17f93406a91e9658c39d4a47668893a9336a",
        "scene_archive_sha256": "c42ff87612735847a91061267df356f3ad6d45bab2ec4025db5c9e5cf7ff3a61",
    },
}

# SHA256 of the supplied inference files and original dataset members.
MODEL_SHA256 = {'ex4dgs': {'coffee_martini': '976901bab79b6b2e56013ce847b45951cd6b14c30204f101e91b6613a54b9e95',
            'cook_spinach': '5171c147c5790d62df93f409741243953eff34daf5dc1ab0f2afc359eb8c52b8',
            'flame_steak': '49031f71f1d15d89cc6ca43289c98347791bf9ea15ad08e294298713871f248b',
            'sear_steak': '1d84ef0fcde93f181d9b9bd084caac0590e74250ae0ad10445834f51b08a1eff'},
 '4dgs1k': {'coffee_martini': 'b0ed670228a6436ea2901fbfd2c138d43e3ef4ac086b0700d9f99624c2bd1578',
            'cook_spinach': '8be8cba5e8e2fdd7f8fc37b11788f8a2768c8f07e660b9532bff2b7d7dcc2ea8',
            'flame_steak': '332b718a2302f4828a01b1284f6ae2c89d2d80efd7d602f9aa0ced21bcd17e36',
            'sear_steak': 'd8597e8308c1557e5a4766efc2d42a153bf31bbbdfb6f5aec1e41ca588f9b511'},
 'anchored4dgs': {'coffee_martini': '0490dd70994f29f629f6b49e1e3129afb7a1b6de5ec4319d40c4d5ab907bddc8',
                  'cook_spinach': '21cb84215834eb6d5821486ac62cfd73f15b0d8d75aacddeb2de71428f014368',
                  'flame_steak': '13faf386771d19cda24318d4d02f7c73b26c3c3749994383b78425defefbc8e3',
                  'sear_steak': '8ccd72fefeb71b75d493d239fbcc03e6954c47d127e728d4732c703503cee429'}}
DATA_SHA256 = {'coffee_martini': {'cam00.mp4': '09cc8b695b78ef13d7e02a3a8a1e7e4f5a9f915730c4515c7cc9bcd06c843ad9',
                    'cam01.mp4': 'e508e9d36e7456b62baed1c1dca1b329cd6c3dd552df5b8d23a350df7f57e396',
                    'cam02.mp4': 'f23ad58769f22343980b9d82f5e922b5edce30b77eea954543e35969edb418bf',
                    'cam04.mp4': '73eb3314cd12bf59a271afa267826e1b70c581794477c2736c6ee3a37a9ef7b4',
                    'cam05.mp4': '68775596b87f8e5bd77173c9095030db0600310bb02f308e8a21f67453fd1346',
                    'cam06.mp4': '3a5cb2acd5266d23c83d7dbd16c10b69da10eac73b689011e275b0da33695f0d',
                    'cam07.mp4': '391082602f37d5a1e7a0c5c1aee04e4a6b3153e95d27703f975f4d2e8b6780aa',
                    'cam08.mp4': '71d8d9bc6e765f8e8bca818bf1d51b210d0764142106941a057cb1f3faeb0e16',
                    'cam09.mp4': '18c63de86174a3afd4db6dc9b5ee089bf5c92f9e74b4c343f9b118ebab804dcc',
                    'cam10.mp4': 'fce45ee769028d36410ee1afb721eebd554861eaefcee4642a881dd2e717245d',
                    'cam11.mp4': '290bff5a7d953598c8a0316b39d2ad941f496f07d3591a11babc98f54629e72b',
                    'cam12.mp4': '9ecec9603c052621950fe42ee10bac0e00d7fb68113947c633a1d7b8c61bde9d',
                    'cam13.mp4': '02e1d56a8af8caf1141a614e6bc5bc8b6fe278cd7fc07730d2f65363a9761314',
                    'cam14.mp4': '0055e3eafbf9737016aebd13c6c206b0b6d3a095954ec9e9adb5c5614114d7d9',
                    'cam16.mp4': '696fccec78b0ac9850875b945aa9e3f6a6e41f7c6af6a96e276222d0ac0834b2',
                    'cam18.mp4': '1ceab97346ad9d7dbe256b16e21f3421a93675b5783c1ec7f8b8e4c7f8dd9f01',
                    'cam19.mp4': '8bd59da8e1c80f88937e288df824e334c5a4235f2141dc73a66a73102b0cb799',
                    'cam20.mp4': 'a9338edce43fa19ad491d36dcd8f436de45d168b21c94c7a96f5e01b5164451c',
                    'poses_bounds.npy': 'b82479fe530b6bf6792c718276aa69bd499a5ee777491eb5da66121c4247cf50'},
 'cook_spinach': {'cam00.mp4': '4b32446a3950366669f0639c90390c6a69bc39eef85b70194311f42e1c30e67f',
                  'cam01.mp4': '94acbc4475a1d5584606603fbc54fb50974a412408d445842d1cf22407b48ece',
                  'cam02.mp4': '66dde1fa80095b8646722533e6cc26effa383c66fa5235fecf14ea0a64ffb980',
                  'cam03.mp4': '1755fc9ac3efffe71e6d005855b9c4220f5dba7791dd3aed6369c64075997861',
                  'cam04.mp4': 'a641ec6351ef1aeb77e88c5cc8636cf7594a77a5bf5c7c860917b4dec29b1c50',
                  'cam05.mp4': 'a13951bfcd15480cd48e686c31c193a1ee947c697bd46f4a199d82c02baac87f',
                  'cam06.mp4': '7e6bc49c6db9f3c66a29e4a0609f514e50464621a94548e5100f02d116d65db9',
                  'cam07.mp4': 'af5ff160c197d32f0435b52a996d42b4e6ccda5b5633c14b1b455c717d4f90c6',
                  'cam08.mp4': 'b43ca38c7edccaee9b2f4dda6aa982b0f871d43e815d65007dd650bef616bc37',
                  'cam09.mp4': '54b5c71f3e8973b921fdf84c83aadfd8849d8249e448ec82d3e17881e9216a8e',
                  'cam10.mp4': '1205a35a38214c82fc36982a1ab99061a0d547066507cf0ed61ceb192f71f000',
                  'cam11.mp4': '289be7be4d7e01e3286c14761f0ada2ed4f045f654a073b7dd2b5ab6f284715b',
                  'cam12.mp4': '5eaff0fa3c903716224fcfbfe95f7173867709039c818165ffdd5ea118d2ef37',
                  'cam13.mp4': '59d5cc6808c2fa1268c0d8f85f216d2e251169b105a01270672dd345e0645b35',
                  'cam14.mp4': '764c41cfed8d2513e76bed93e127ecaa4ce81da27084e625088bdd0117ff15f8',
                  'cam15.mp4': '30b62cd2923310ebaedea2217a5c49e086278b7df83d1cc586428928911ded7f',
                  'cam16.mp4': '71b4f5b7d3869b498639dbd998c643936976bf15c8abcf87cddd11e090c8aee1',
                  'cam17.mp4': 'ec1363b16aa19bf38d7eb6478137ea91033fabf4b8aa8ad0fc1fa8e4835f3acd',
                  'cam18.mp4': '6a84e57d6f3110048038b708535d3fe65c622fd2971760f95c114af871d5b520',
                  'cam19.mp4': '34bbdb5ca3f7cd5cfda9e0eb5d18a8f319f35d3becb5accf7ca416b37cc02df9',
                  'cam20.mp4': 'a6402310a6fa60f734f7a8a20e06f48cb7582ddf094900d53bd039dc42fb8090',
                  'poses_bounds.npy': '5479315fff3c896cab42dc322b3ba89c96d56b8ad83284482cefea61ff2b1309'},
 'flame_steak': {'cam00.mp4': 'c0b19854049da7ae3e8c770c65c082d27878dbe99d118cb58b1f3bcb9dd580c5',
                 'cam01.mp4': '2e3b4c6ef933f72c658c699c52145e4ff300bf6c4fa46d79664c3d802b2c62e6',
                 'cam02.mp4': 'd884fcff283cbbba4e3a790e3afe2fef4a08569ffeb146855450e734c2955490',
                 'cam03.mp4': 'b45cec5ae99cd2386aaf0b0a6821ec21f6ba17d1855be04bee6f333c849f6a18',
                 'cam04.mp4': '959b5a302b5a37a6db25ebfa982947879c798928ff873abc00ccb25de6a49d5e',
                 'cam05.mp4': '9ee38f5d41271c5ae73afab837743f4aed57490ddd6f4ee43d37ee81a271d678',
                 'cam06.mp4': '2552596063a2a654ca605cd0caac087e2d84cd0ce200061810cb0c133f9f1645',
                 'cam07.mp4': 'b2318a6fa76aace4ed122e62c0b077b2e3f34d62489c432450c749439a616e03',
                 'cam08.mp4': 'babed972d0c48dabe403eeab085ca4023561b0f9aa289b2c0db8bd7cf04ee6f1',
                 'cam09.mp4': '27124de6e5930b6c935af21e67207f295099cfd45002fedde2341153bc867afd',
                 'cam10.mp4': 'b63bc716b39ddc96be0040e2e464e7f36601e4a0bb4aa75409925450f780596d',
                 'cam11.mp4': '919c521a76492a86533c3760591bb855991135d81d0f17d1c8eaca3057954037',
                 'cam12.mp4': '4a4b47ebe14a623ee565571a8434784d517d65c2011f885e85197ab5484810fa',
                 'cam13.mp4': '758e2695c005e5058534cdcbbe50bf97ffbc3cd858fce6d9255c18c8d8a8b86e',
                 'cam14.mp4': 'b2137e6662ef253c5839a371c467cf9561c0ce88ce92b7e18343022cfc304ec5',
                 'cam15.mp4': '222a2215f4d840cc43c2f203fcdf0b32945865b1193fb37bdb6299560cfc91e6',
                 'cam16.mp4': '2f7c5ee7833d6edbf24dc0b453debfabeaea6ac4bd3952e7225af0717d5b091e',
                 'cam17.mp4': '8fd98b0548c980401a3228c7d822fe40e622ea8bd52286be05314401eae8dd0e',
                 'cam18.mp4': '2314e852cbd121a02ae8676f8d4bf7b8503754fe3480f73edb77bf834268ba4b',
                 'cam19.mp4': 'dfa8a77772a53e2bd811473fbdbd0049ee8b04d33b14670ac734a49e60f3c0a4',
                 'cam20.mp4': '69e8b7a9e8f4d181cf5fce4625df2d8945a1e4253b7ad8fd08a978e074a41f62',
                 'poses_bounds.npy': '42e7e6d8d44f27f6ee3fb2b472ee0051a73c3a3387003ccd453c80567bb766a8'},
 'sear_steak': {'cam00.mp4': '2431743fb2eaf2a03494c3f6653d5b8a11d60ddb58f0392652c12ac0c15ee416',
                'cam01.mp4': '1cc3cf78ce2d38a3467b5f040d64b90499e2eab7d3a91fa7aa1b8a8a1671d9f7',
                'cam02.mp4': '98e7cb3919356709b969744bf270d6970b4f0f494ea1b3667971dbd2ccc8f75b',
                'cam03.mp4': 'c25c500b7f5440850b7ec51b06ba4d4c5f522df0275ae3ac16486466863bfd93',
                'cam04.mp4': '4e04b41081a21569f87e466efcf798f60f180439d14b9804683728523c14c352',
                'cam05.mp4': '35a8dc5b2213279cd8648787c69d79e4865a6eff7d992085c03a3dc3ced6ac8e',
                'cam06.mp4': '220ad75fc666815544984be071033bd8ce8809533b102b2c9cbe9d1d7a129c69',
                'cam07.mp4': '7f0d2d0bc10d835d4cc3a27a764ae237dfd19320b4d2fab593ba9e534e85498a',
                'cam08.mp4': 'e2092d48f65f1e4368633dca1773cb50b6c0a39bf16cbca1ce7fc1cf69f9c234',
                'cam09.mp4': 'c01716e3c1f74ea3db2f55da52bfc45dc9455aa785a5d1cb546fbaf5b9e245f9',
                'cam10.mp4': '2d730eb55dfb7fd97127f4d95518193d920d732f9b45d4e1b90d91f95f88ba66',
                'cam11.mp4': '2fe7e94ac4b3161fe50f4307cc0be87eeb4900528b6a93ae89e032f11e83c8bb',
                'cam12.mp4': '907f818a2b10727725c9b03ccef0554f1c46c6c5c871531744ed4f892c97ae68',
                'cam13.mp4': '364ee23ec72e4392606e2bad251774559c7e1376776f8744ce031948c7bc20aa',
                'cam14.mp4': 'd0493c0d3870bd85673ad876b7101ae282f98b6602cca0a99cd0341eb1fa1b04',
                'cam15.mp4': 'c71de047fb7cc41ba1f67cc0406ed1b6f644b70b9d541c882ce16eef2197a8e6',
                'cam16.mp4': 'eccedb924b5a30c4d744dd315d0aed2f9d835adbc8c30c2a656c76955b767c73',
                'cam17.mp4': '836d2da13438c1682db54e72c7fd35a19cd0e0f460ad5f392174072c23d98f3f',
                'cam18.mp4': '7cdbcdcf4cbfa10cfafb88d3fd23a4820cef8969d31e4cde28f4959a468392b3',
                'cam19.mp4': 'ff613fabc488fcd7d8316eb29ff1e63db36db08a0ba9910e530c2c39eeca3b7c',
                'cam20.mp4': '0a3c26c642437f8780cd7f0d25dea761df6317ef8be50df95777e75bd33287b2',
                'poses_bounds.npy': 'e436726227fbdcab1095aa326b98cfc32be5da535a9137b9ed157e9a1640e465'}}

# Checkpoint digests for the two official-camera inference inputs.
MODEL_SHA256["ex4dgs"].update({
    "dance3": "9efd284ebac2a10a0ab08f0e82443c78c444c701978cd14ef352491bd2d2d743",
    "actor1_4": "2b417357d4f4bb7de6e2063d1c231e97df29121b3d747dee1ef8faa3276e70e1",
})
MODEL_SHA256["4dgs1k"].update({
    "dance3": "1af70cd4d389785131311713baa39ca1382b2b6a97ef747b7554fad20b2e709b",
    "actor1_4": "3b0d1e70042940e402521dd609b5a5c371b6d68bdc3efb50323d5519b076788f",
})
MODEL_SHA256["anchored4dgs"].update({
    "dance3": "2851551556dfbfe50ac0b30058235086636c6469eb0c0f36002aeecc06a77e33",
    "actor1_4": "3c93325c5136c6333b416b209bb923b23a324f5508f0aae0ffb01479961e14c2",
})

MODEL_PARAMETERS = {
    "4dgs1k": {
        "dance3": {"source_iteration": 8537, "finetune_steps": 1970, "frame_count": 300},
        "actor1_4": {"source_iteration": 7495, "finetune_steps": 3265, "frame_count": 200},
    },
    "ex4dgs": {
        "dance3": {"release": None, "iterations": 2837, "duration": 300, "resolution": 1, "near": .2, "background": 0},
        "actor1_4": {"release": None, "iterations": 2720, "duration": 200, "resolution": 1, "near": .2, "background": 0},
    },
    "anchored4dgs": {
        "coffee_martini": {"training_steps": 658, "source_iteration": 21500, "frame_count": 300},
        "cook_spinach": {"training_steps": 803, "source_iteration": 22500, "frame_count": 300},
        "sear_steak": {"training_steps": 1459, "source_iteration": 19500, "frame_count": 300},
        "flame_steak": {"training_steps": 1110, "source_iteration": 7500, "frame_count": 300},
        "dance3": {"training_steps": 1939, "source_iteration": 8537, "frame_count": 300},
        "actor1_4": {"training_steps": 1850, "source_iteration": 7495, "frame_count": 200},
    },
}


@dataclass(frozen=True)
class Workload:
    name: str
    directory: str
    extension: str
    images: str
    image_prefix: str
    source: str
    parameters: tuple[tuple[str, object], ...]

    def checkpoint(self, scene: str, root: Path = ROOT) -> Path:
        return root / "checkpoints" / self.directory / f"{scene}.{self.extension}"

    def image_directory(self, scene: str, root: Path = ROOT) -> Path:
        return root / "datasets/n3v" / scene / self.images

    def sha256(self, scene: str) -> str:
        return MODEL_SHA256[self.directory][scene]

    def parameters_for(self, scene: str) -> dict[str, object]:
        return dict(self.parameters) | MODEL_PARAMETERS.get(self.directory, {}).get(scene, {})

WORKLOADS = {
    "explicit4d": Workload("explicit4d", "4dgs1k", "pth", "images", "cam00_",
        "https://papers.neurips.cc/paper_files/paper/2025/hash/6c39d1a7eecfd98570d74bf7efec1be7-Abstract-Conference.html",
        (("retained_fraction", .2), ("keyframe_stride", 20), ("volume_power", .1),
         ("sh_degree", 3), ("sh_degree_t", 2))),
    "ex4dgs": Workload("ex4dgs", "ex4dgs", "zip", "cam00", "",
        EX4DGS.url, (("release", "v0.1"), ("iterations", 40000), ("duration", 300),
                    ("time_interval", 10), ("time_pad", 2), ("var_pad", 3),
                    ("resolution", 2), ("sh_degree", 3), ("near", 4.0),
                    ("far", 300.0), ("kernel_size", .1))),
    "anchored": Workload("anchored", "anchored4dgs", "pth", "images", "cam00_",
        "https://doi.org/10.1145/3757377.3763898",
        (("feature_dim", 32), ("hidden_dim", 32),
         ("bindings", 10), ("tau_t", .05), ("growth_decay", .1),
         ("timespan_quantile", .5), ("spatial_voxel", .001), ("temporal_voxel", .033),
         ("static_after", 1500), ("opacity", "tanh"), ("covariance_input", "feature"),
         ("color_input", "feature_view"), ("covariance_scale", "base_times_sigmoid"),
         ("static_geometry", "mean_at_temporal_center_no_temporal_decay"))),
}
# Training source and final steps of the supplied compact explicit4d files.
EXPLICIT4D_STEPS = dict(coffee_martini=(20000, 21500), cook_spinach=(21000, 22500),
                        flame_steak=(6000, 7500), sear_steak=(16500, 19500),
                        dance3=(8537, 10507), actor1_4=(7495, 10760))


def combinations(workload="all", scene="all"):
    """Yield the selected workload/scene pairs in default execution order."""
    for name in WORKLOADS if workload == "all" else (workload,):
        for selected in SCENES if scene == "all" else (scene,):
            yield WORKLOADS[name], selected
