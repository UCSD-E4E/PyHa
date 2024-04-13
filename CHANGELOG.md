# CHANGELOG



## v0.1.0-rc.2 (2024-04-13)

### Fix

* fix: add wheel to build ([`2d828e0`](https://github.com/UCSD-E4E/PyHa/commit/2d828e0b13c21dbe1afc282e7d769e2249c440ea))

### Unknown

* Merge branch &#39;test-python-package-prep&#39; of https://github.com/UCSD-E4E/PyHa into test-python-package-prep ([`c532a1b`](https://github.com/UCSD-E4E/PyHa/commit/c532a1bf124fbb2183c4c75c1858527534512121))


## v0.1.0-rc.1 (2024-04-13)

### Ci

* ci: Fixes env test to only execute on env changes ([`8cb235d`](https://github.com/UCSD-E4E/PyHa/commit/8cb235d0f5812bee0437ba6ab937ec1422e34481))

### Feature

* feat: added semantic releases ([`d23038b`](https://github.com/UCSD-E4E/PyHa/commit/d23038b3fcceb400998e37ec9d328afa4f8734c0))

### Fix

* fix: again attempt to get sv to run on test branch ([`c397780`](https://github.com/UCSD-E4E/PyHa/commit/c3977801e096f0915da28be5dc6bfa059a9a70a8))

* fix: test semantic versioning on test branch ([`6c87605`](https://github.com/UCSD-E4E/PyHa/commit/6c87605d22cfc7e8bd39f93b4f14c4eb3731dfad))

### Unknown

* fix: ([`1a0924f`](https://github.com/UCSD-E4E/PyHa/commit/1a0924fb9c91a19c1013018f00ef0de21afd1227))

* Update pyproject.toml

feat(package): add semantic verisioning ([`19616c5`](https://github.com/UCSD-E4E/PyHa/commit/19616c554ea6fb60a272459b4b17ad8aa810807f))

* Update create_release.yml ([`70f87fc`](https://github.com/UCSD-E4E/PyHa/commit/70f87fcb1bb9e0e57622e9bf7fffd1e6850edb41))

* Update create_release.yml ([`80111ef`](https://github.com/UCSD-E4E/PyHa/commit/80111efe0284b00c875a0f2994d1a2c7a4c0dace))

* Update create_release.yml ([`10aebc4`](https://github.com/UCSD-E4E/PyHa/commit/10aebc477b33a11a1da4fdca3223e7dac4055acc))

* Update create_release.yml ([`63f8f03`](https://github.com/UCSD-E4E/PyHa/commit/63f8f0354c4ff5db8e085a92523c51e09fd1fda8))

* test creating a wheel ([`41a959a`](https://github.com/UCSD-E4E/PyHa/commit/41a959a59d95e4eca9dc95e94f72a28bae6cfe99))

* testing releases ([`12f7939`](https://github.com/UCSD-E4E/PyHa/commit/12f79390e000cdaa24b5b520a4928fa2a46344da))

* Update test_gh_cli.yml ([`827db1d`](https://github.com/UCSD-E4E/PyHa/commit/827db1de16ee950af8d6070c4b26b1a11ab6a85a))

* Update test_gh_cli.yml ([`51ce06e`](https://github.com/UCSD-E4E/PyHa/commit/51ce06ee68e3d3105f0e40ca430e76d81d3a2257))

* Update test_gh_cli.yml ([`19cca58`](https://github.com/UCSD-E4E/PyHa/commit/19cca5836a211b09abc4b3ba6a7b3b2cdbf7889f))

* Update test_gh_cli.yml ([`bab6a07`](https://github.com/UCSD-E4E/PyHa/commit/bab6a078ecdd41d73790de4db22704927006e9bb))

* Update test_gh_cli.yml ([`8254e6d`](https://github.com/UCSD-E4E/PyHa/commit/8254e6d7fe9877e1ca2424c355cf7674fa38cecc))

* Update test_gh_cli.yml ([`5332f21`](https://github.com/UCSD-E4E/PyHa/commit/5332f21e51edd33d7ac6fa99f4b61ce6876d7dac))

* Update test_gh_cli.yml ([`aabc365`](https://github.com/UCSD-E4E/PyHa/commit/aabc365cfe35602e3a3b7f3d2dbff26b187b4f03))

* test auto create issue ([`55dcf9a`](https://github.com/UCSD-E4E/PyHa/commit/55dcf9a6711c01bdac36ccc7aa359580990d0f29))

* test build a wheel ([`8722227`](https://github.com/UCSD-E4E/PyHa/commit/8722227b78e6b06fa8bbaea70b4483948939c22d))

* Poetry Environment (#159)

* feat: Adds poetry
* feat: Adds workflow
* feat: Adds testing
* chore: Removes conda
* Replaced .append with pd.concat for pandas upgrade

Co-authored-by: Nathan Hui &lt;nthui@eng.ucsd.edu&gt;
Co-authored-by: Sean Perry &lt;shperry@ucsd.edu&gt;
Co-authored-by: TQ Zhang &lt;116205967+TQZhang04@users.noreply.github.com&gt;
Co-authored-by: Samantha Prestrelski &lt;samantha.prestrelski@gmail.com&gt; ([`c991cda`](https://github.com/UCSD-E4E/PyHa/commit/c991cda085f8018b0c371babe6a8e4a3b7bf3b2f))

* update 0 predictions in tweetynet, fix error messages (#157) ([`4b27fa6`](https://github.com/UCSD-E4E/PyHa/commit/4b27fa6a2e535b515a42bb0529f428fc903507db))

* Add torch to conda installation (#156)

Torch was not listed on Conda yaml when installed
Ran `pip install torch` to install torch and cuda tools
Exported to `environment_Ubuntu20.04.yml`
Copied the changed pip dependencies into other environments
Confirmed that Ubuntu20.04 environment worked by recreating env
Other environments should work accordingly ([`b3e3e43`](https://github.com/UCSD-E4E/PyHa/commit/b3e3e4335d479aee2f40d232323ed8d12d0c48d9))

* Merge pull request #150 from UCSD-E4E/ctrl_c_exception_handling

Ctrl c exception handling ([`858714b`](https://github.com/UCSD-E4E/PyHa/commit/858714b077b00c30576a11ad386f24bdd9a9097d))

* Delete unnecessary .swp and .un files ([`f317877`](https://github.com/UCSD-E4E/PyHa/commit/f3178776c9e67399e87ce7b3c82180d15c6248aa))

* Delete unnecessary copy of IsoAutio.py ([`c0a684a`](https://github.com/UCSD-E4E/PyHa/commit/c0a684af08995a58a88ef759cd684980b31549d2))

* Merge branch &#39;ctrl_c_exception_handling&#39; of https://github.com/UCSD-E4E/PyHa into ctrl_c_exception_handling ([`a0e362b`](https://github.com/UCSD-E4E/PyHa/commit/a0e362b3bb727ebf2db3acb014999434fdc233e0))

* Add KeyboardInterrupt to &#34;analyze&#34; for BirdNET

also only import exit from sys in IsoAutio ([`417de3a`](https://github.com/UCSD-E4E/PyHa/commit/417de3a6c32eaa4f4e94a52a4c40e88c977e904e))

* remove pycache files and reorganize gitignore

https://stackoverflow.com/questions/56309100/how-to-ignore-the-same-name-directory-pycache-in-a-project ([`cc82dab`](https://github.com/UCSD-E4E/PyHa/commit/cc82dab360d9526104102a49bc41a650fa24852f))

* moved KeyboardInterrupt before BaseException

just for the only try-except block that needs it. ([`f85d2ba`](https://github.com/UCSD-E4E/PyHa/commit/f85d2ba3c897cca676b17917503b777b8d28a420))

* Added verbose key to tutorial isolation parameters ([`1bbf05e`](https://github.com/UCSD-E4E/PyHa/commit/1bbf05e7caa0f89c5796f5896f5f2143f4f637b6))

* fixed missing verbose key in jupyter notebook ([`5673e2e`](https://github.com/UCSD-E4E/PyHa/commit/5673e2e60d1f725e5f72718aea845959dbfcaa19))

* Added import sys
- also noticed a resampling typo that was deprecated due to the use of resampling rather than just downsampling ([`c22a9c8`](https://github.com/UCSD-E4E/PyHa/commit/c22a9c8c01108d1e01b32dd293287a9ed37e2b74))

* Merge branch &#39;main&#39; into ctrl_c_exception_handling ([`e72e8e2`](https://github.com/UCSD-E4E/PyHa/commit/e72e8e20a7ad58e9cd037257bbc9693e647da525))

* Moved KeyboardInterrupt exceptions

Moved KeyboardInterrupt exception before BaseException. Also added second &#34;r&#34; in &#34;interrupt.&#34; ([`cef7156`](https://github.com/UCSD-E4E/PyHa/commit/cef7156927995eb35c2757da9c5ed6885b601a02))

* Keyboard interupt &amp; checkVerbose assertions

KeyboardInterupt exceptions added to try-except blocks.

addition of assertions for checkVerbose ([`fcd8627`](https://github.com/UCSD-E4E/PyHa/commit/fcd862721b37e884b35caf3de64a387478da9127))

* Update IsoAutio.py

Added assert statements to checkVerbose function. ([`0f2ccb6`](https://github.com/UCSD-E4E/PyHa/commit/0f2ccb6d0949466a71093465a41bd3f5c0b89c2b))

* Delete PyHa_Model_Comparison.ipynb ([`5d7abcd`](https://github.com/UCSD-E4E/PyHa/commit/5d7abcd7b82bf387f5ae873baacc817e5fbd53ba))

* Delete PyHa_Testing-Copy1.ipynb ([`d6792dc`](https://github.com/UCSD-E4E/PyHa/commit/d6792dc7bf2229b870baa8876be5c131eacf936c))

* Delete PyHa_Testing.ipynb ([`994b868`](https://github.com/UCSD-E4E/PyHa/commit/994b868de639f173d5ac2234cf9ce231c3e8e8b9))

* Merge branch &#39;main&#39; of https://github.com/UCSD-E4E/PyHa into main
- Fixed merge conflicts ([`4ef0c23`](https://github.com/UCSD-E4E/PyHa/commit/4ef0c23292c4d9bc9d7c491d6b86d0403858bf22))

* Merge branch &#39;type_assertions&#39; into main ([`4b5ae03`](https://github.com/UCSD-E4E/PyHa/commit/4b5ae03a3c60380f7c170551bb881c48f9e7b340))

* Fixed birdnet demo isolation_parameters
- Confidence was too high causing statistics methods not to run ([`3a890c8`](https://github.com/UCSD-E4E/PyHa/commit/3a890c82ba6f463404cbefd37a4a9955689fb3ee))

* More assertion checking
- Added in type checking for IsoAutio.py file
- Fixed some mistakes in prior commit on visualizations.py
- Changed input paramater name of spectrogram_visualization&#39;s &#34;automated_df&#34; to &#34;build_automated_df&#34; ([`3fc4008`](https://github.com/UCSD-E4E/PyHa/commit/3fc40087b19b223dddaa6fac2a37d7d4ec74bcbc))

* More input assertions for visualizations.py ([`d29bd8f`](https://github.com/UCSD-E4E/PyHa/commit/d29bd8f7fe7a606c3219400da49ca35385d19ef2))

* Added in assert statements for functio input validation
- Completed the statistics.py file
- started the visualizations.py file ([`9d09e4d`](https://github.com/UCSD-E4E/PyHa/commit/9d09e4d171e734957fcb46a0cf6f02db4179c3ec))

* Added a verbose toggle feature (#139)

Co-authored-by: ayao &lt;anthony.anthony.yao@gmail.com&gt;
Co-authored-by: RyanHUNGry &lt;ryhung@ucsd.edu&gt;
Co-authored-by: Samantha Prestrelski &lt;33042752+sprestrelski@users.noreply.github.com&gt; ([`3593b52`](https://github.com/UCSD-E4E/PyHa/commit/3593b52810e462c11cb018c7ea89185bfe3d5743))

* Merge pull request #140 from UCSD-E4E/ReadMe_filesize_reduction_installation_option

Update README.md ([`3ebcf97`](https://github.com/UCSD-E4E/PyHa/commit/3ebcf970303cdc1b973e91cc41f5234de791c0be))

* Fixed typo

Accidentally added a space between -- and depth in the git command. ([`afa62e0`](https://github.com/UCSD-E4E/PyHa/commit/afa62e0395478b2f1f2963f52c535c64c5f7fa98))

* Update README.md ([`ff13c8c`](https://github.com/UCSD-E4E/PyHa/commit/ff13c8c48f5c0fd78d6ea3f4af1c9f057f287772))

* Update environment_Windows10.yml ([`d336cab`](https://github.com/UCSD-E4E/PyHa/commit/d336cab5ff350408c5f72dd17e31d8ff4e349445))

* Optimizing isolation techniques (#124)

* Optimizes the Steinberg, Simple, Stack, and Chunk isolation techniques with numpy
* Roughly 5x faster method of Steinberg isolation

Co-authored-by: pranjalc1 &lt;86537904+pranjalc1@users.noreply.github.com&gt; ([`0c4ef29`](https://github.com/UCSD-E4E/PyHa/commit/0c4ef2941a745f2e611ae5ca52dcd7196a8f7736))

* Generalizing file reading with librosa (#122) ([`b58b13e`](https://github.com/UCSD-E4E/PyHa/commit/b58b13e810a53c4352d3be8bece18dee818646fa))

* Increase efficiency of IoU scores (#15)

Refactors clip_IoU method with linear algebra and numpy ([`fa91052`](https://github.com/UCSD-E4E/PyHa/commit/fa9105298471a687b6d0eaf95d9a4e3cd626903f))

* Fixing TweetyNet typos (#131) ([`40a61e0`](https://github.com/UCSD-E4E/PyHa/commit/40a61e02954bb7af2249341b9cafbb3b9f37592e))

* Label chunker (#114)

* Added label chunker
* Added documentation for the annotation_chunker
Co-authored-by: shreyasar2202 &lt;ars.shreyas@gmail.com&gt;
Co-authored-by: Sean Perry &lt;shperry@ucsd.edu&gt;
Co-authored-by: Samantha Prestrelski &lt;samantha.prestrelski@gmail.com&gt; ([`2c48a85`](https://github.com/UCSD-E4E/PyHa/commit/2c48a859460472743aa6e287582f244dffdebd23))

* remove notebooks (#121)

* .gitignore fixed and removed notebooks

* remove notebooks ([`b27cfc6`](https://github.com/UCSD-E4E/PyHa/commit/b27cfc6ad87b51ac0fb7c6a817bd23651b7d78b9))

* .gitignore fixed and removed notebooks (#120) ([`675bec0`](https://github.com/UCSD-E4E/PyHa/commit/675bec011022f1725510b5f03704241554bceebc))

* Conform tweetynet to recent PyHa refactor (#113)

* Added TweetyNet model to PyHa
* Added TweetyNet local score and spectrogram visualization output
* Added new conda environments for MacOS and Windows 10
* Added testing notebooks
* Optimize panda use on Steinburg isolate technique
* Updated .gitignore for cached files and testing
* Updated README, Tutorial Notebook, and documentation
* Improved error messages

Co-authored-by: mugen13lue &lt;mugen4college@gmail.com&gt;
Co-authored-by: Samantha Prestrelski &lt;samantha.prestrelski@gmail.com&gt;
Co-authored-by: Vanessa-Salgado &lt;vsalgadozavaleta@ucsb.edu&gt;
Co-authored-by: Sean Perry &lt;shperry@ucsd.edu&gt; ([`b6b24b9`](https://github.com/UCSD-E4E/PyHa/commit/b6b24b944261a15e010cca64c8fbe6c785f4c709))

* Merge branch &#39;TweetyNet_integrations_2_main&#39; of https://github.com/UCSD-E4E/PyHa into TweetyNet_integrations_2_main ([`4763ee5`](https://github.com/UCSD-E4E/PyHa/commit/4763ee54a4cc6605f002bbb2ca4a17da33a7aef8))

* Update .gitignore

Updated .gitignore for cached files and testing
Optimize panda use on Steinburg isolation technique
Improved error messaging on Pyha ([`2509e74`](https://github.com/UCSD-E4E/PyHa/commit/2509e747f048ad7b174423b3c4e81f4f36d34afc))

* Update .gitignore ([`b376ce5`](https://github.com/UCSD-E4E/PyHa/commit/b376ce5cdc79b99050a7fd862f25057133ab377a))

* Update .gitignore ([`a67d9a5`](https://github.com/UCSD-E4E/PyHa/commit/a67d9a56db8d8262c01f0730f70dbe9d1fcd38bd))

* Improved error messaging on Pyha ([`0690ab9`](https://github.com/UCSD-E4E/PyHa/commit/0690ab9c79b75cd623ca0751ae39a4718d389964))

* Cleaned up steinburg code ([`08d5170`](https://github.com/UCSD-E4E/PyHa/commit/08d51700c8bdcafa9976699011d31a3caaf9a521))

* Fixed bug with new steinburg fix ([`6f2da05`](https://github.com/UCSD-E4E/PyHa/commit/6f2da0596764637cfff9a4e84284c5e0fa65bc89))

* Optimize panda use on steinburg to fix bug ([`5f00a09`](https://github.com/UCSD-E4E/PyHa/commit/5f00a09ae015e8b3cfc48541ac6d170dc409e45e))

* Fixed gitignore ([`eeea93c`](https://github.com/UCSD-E4E/PyHa/commit/eeea93cd59b492304e73658caed17f49b01acb40))

* Uncommented old normalize code ([`b3846ff`](https://github.com/UCSD-E4E/PyHa/commit/b3846ffcff2229b8a6bc012d3914b0460da6e5d9))

* Merge branch &#39;TweetyNet_integrations_2_main&#39; of https://github.com/UCSD-E4E/PyHa into TweetyNet_integrations_2_main ([`62f2729`](https://github.com/UCSD-E4E/PyHa/commit/62f2729f0d92a542ab63b3999a907e53eb9f0051))

* Use tweetynet normalization ([`b0aca41`](https://github.com/UCSD-E4E/PyHa/commit/b0aca418389d1ca1884d0e32e51f1664757710c1))

* documentation update, renamed testing scripts ([`b937c2e`](https://github.com/UCSD-E4E/PyHa/commit/b937c2e4eedfc835b291f0635ba009ba5b8f35dc))

* Removed testing folder because that was a bad idea

It boke a bunch of imports, will have to try again with a different idea later ([`a9a9f7f`](https://github.com/UCSD-E4E/PyHa/commit/a9a9f7f68e2d63300ffba5210cf610a2bea0731f))

* Merge pull request #119 from UCSD-E4E/TweetyNet-Testing

Added Testing Code For Tweetynet / Docuemtentation Updates ([`e3a25d8`](https://github.com/UCSD-E4E/PyHa/commit/e3a25d8db1b8b9a0ac8630526be8d04f079f9f0e))

* Merge branch &#39;TweetyNet-Testing&#39; of https://github.com/UCSD-E4E/PyHa into TweetyNet-Testing ([`4dbef91`](https://github.com/UCSD-E4E/PyHa/commit/4dbef91006a6d1dc22469a2e6b0b94e9c3ffee47))

* Moved testing scripts to testing folder ([`20c1536`](https://github.com/UCSD-E4E/PyHa/commit/20c15362a70682c07cdfc1051173a38bf7b7ec03))

* Merge branch &#39;TweetyNet-Testing&#39; of https://github.com/UCSD-E4E/PyHa into TweetyNet-Testing ([`d1b2fbd`](https://github.com/UCSD-E4E/PyHa/commit/d1b2fbda00b40194099db053a6164a21be27eef2))

* added tweetynet documentation ([`b113ac1`](https://github.com/UCSD-E4E/PyHa/commit/b113ac1b5cdd3a113617f97db0083baa518e5897))

* Fixed spectrogram_graph visualization to readme ([`2d672f0`](https://github.com/UCSD-E4E/PyHa/commit/2d672f06fa082a00dbab7efbf942f95bd2f02ff3))

* Added chunk isolate to testing suite ([`5f6ce9f`](https://github.com/UCSD-E4E/PyHa/commit/5f6ce9f9d0664bd1c6a17c8795673a3a60e0436b))

* added model comparison and spectrogram vis testing ([`6ca7ff4`](https://github.com/UCSD-E4E/PyHa/commit/6ca7ff439a12575023587c1bd45145ebb8a9fd50))

* Fixed FutureError warnings

For tweetynet, fixed the liborsa warnings output by adding adtional augrement ([`ca4d4f5`](https://github.com/UCSD-E4E/PyHa/commit/ca4d4f5886019b446ce24916ba2021a830e1b589))

* added batch testing ([`361d27d`](https://github.com/UCSD-E4E/PyHa/commit/361d27dbd915fa6914032c86395cd91bf4932535))

* Meant to be in the conda_environments folder ([`72a2dec`](https://github.com/UCSD-E4E/PyHa/commit/72a2dec12917781bc273d1fea90fc22a0ab567bd))

* Updated Most Recent Ubuntu Conda Environment with PyTorch

- PyTorch is required for TweetyNET ([`7c1a806`](https://github.com/UCSD-E4E/PyHa/commit/7c1a806b64fa5a4e9974a10cfe4a34b13ccc98e8))

* Updated Tutorial Notebook

- Trimmed some unnecessary tweetynet isolation parameters
- Removed unnecessary import of torch. ([`46b5e13`](https://github.com/UCSD-E4E/PyHa/commit/46b5e13cb8623e1df01d464fb76a3503de3b2720))

* Refactor TweetyNET automated label gen/visualizations

- removes the tweety_output bool and moves it to isolation_parameters dict
- updates local_score_visualization() function to reflect spectrogram_visualization() name change and organization
- removes TweetyNET tutorial and a handful of test files
- README documentation for TweetyNet parameters and generate_automated_labels_tweetynet() ([`1ac5554`](https://github.com/UCSD-E4E/PyHa/commit/1ac5554731c434d188c27bb91d5f4181398d8769))

* Merge pull request #116 from UCSD-E4E/minor_bug_fixes

Corrected typos and deleted redundant file ([`f98ba0f`](https://github.com/UCSD-E4E/PyHa/commit/f98ba0f72eda3d83f0ec0e0d0032221231324801))

* Correct typos and deleted redundant file ([`f2873f6`](https://github.com/UCSD-E4E/PyHa/commit/f2873f611f5f33d669620090e6462af966c2dfb7))

* Merge branch &#39;tweetynet_integrations&#39; into TweetyNet_integrations_2_main ([`e55baa3`](https://github.com/UCSD-E4E/PyHa/commit/e55baa3e607155aeeaada4ba0d5899867461fc4c))

* updating notebook with tweetynet example ([`5564bd4`](https://github.com/UCSD-E4E/PyHa/commit/5564bd492c9a404bdeed8f26a76929c89c6e4edb))

* update from microfaune to tweetynet in error message ([`90c0414`](https://github.com/UCSD-E4E/PyHa/commit/90c041451d9cfe24e1fdf3c8ac4a5a8edbed77d4))

* updating gitignore to ignore pycache in tweetynet_package ([`9cda0e7`](https://github.com/UCSD-E4E/PyHa/commit/9cda0e7c3428f967b9fe64a5ebdd83c1510d5391))

* adding environment for windows 10 ([`fb68481`](https://github.com/UCSD-E4E/PyHa/commit/fb6848136694bd30549f49075138cd8583c6dedd))

*  adding conda environment for macOs ([`1be6fea`](https://github.com/UCSD-E4E/PyHa/commit/1be6fea1cf5594e4bcfa851d3d42a195ef749ad9))

* The visualization and creating the automated dataframe works and will now test with various wav files ([`dbe2252`](https://github.com/UCSD-E4E/PyHa/commit/dbe2252f14a34f0fe4e2c1fa8c0334e3ff1177fc))

* Update License

Matching BirdNET-Analyzer instead of old BirdNET-Lite repository ([`97faf00`](https://github.com/UCSD-E4E/PyHa/commit/97faf00bb148c1c2e8e8911253a7d98643eb4295))

* Merge pull request #112 from UCSD-E4E/birdnet_integration

- Birdnet integration
- Statistics Expansion ([`a6e2ea9`](https://github.com/UCSD-E4E/PyHa/commit/a6e2ea98c29c42ef7144813b45fb98879be7617d))

* Merge branch &#39;birdnet_integration&#39; of https://github.com/UCSD-E4E/PyHa into birdnet_integration ([`9996d48`](https://github.com/UCSD-E4E/PyHa/commit/9996d4838fdf3b58779b0c90db03058fc9af23a9))

* Adjusted Tutorial Notebook
- Some markdown didn&#39;t make sense talking about TPs, FPs, FNs, and TNs in the context of the spectrogram_visualization() changes ([`4f3549d`](https://github.com/UCSD-E4E/PyHa/commit/4f3549dc62d3ba1132cf05032cc069f8418a8efa))

* Added Tensorflow 2.8 to windows env ([`6d83f2c`](https://github.com/UCSD-E4E/PyHa/commit/6d83f2c72157b4912daf5e329995bceeb177b39a))

* Fix for visualization typechecking

- Fixed a small bug in spectrogram_visualization() to better handle both bools and dataframes for automated_df
- Added spectrogram_graph(), annotation_duration_histogram() to README ([`1e519c1`](https://github.com/UCSD-E4E/PyHa/commit/1e519c16d0986c5f5cd23ed01c02b58f49cad546))

* Updated License

- Since we are using birdnet source code, we have to reflect their license.
- No commercial use I guess. ([`d4d0592`](https://github.com/UCSD-E4E/PyHa/commit/d4d05922d21f8e53e82f93874d232eea0bdd356d))

* Delete LICENSE

Moving to Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) ([`9504fd5`](https://github.com/UCSD-E4E/PyHa/commit/9504fd56060244dd2ac954d1641422b493b5007d))

* Renamed two visualization functions
- Changed local_score_visualization() to spectrogram_visualization()
- Changed plot_bird_label_scores() to binary_visualization()
- Fixed a small bug in spectrogram_visualization() related to an error warning when a dataframe with automated annotations from multiple clips is passed in ([`de5a9e0`](https://github.com/UCSD-E4E/PyHa/commit/de5a9e0de89648c0005d61920a8aca60c9a4ee95))

* Fixed new Ubuntu 20.04 conda environment placement
- Made sure it is in the conda_environments folder ([`7c873f3`](https://github.com/UCSD-E4E/PyHa/commit/7c873f3abc797da17c3b16a72e2064cf2fa627a4))

* local_scores ability to pass in automated_df (#109) ([`b1d81aa`](https://github.com/UCSD-E4E/PyHa/commit/b1d81aaa21a2c8fa5e1e431f4286814036e25cf3))

* local_score_visualization for non-Microfaune

currently calls generate_automated_labels on the single clip - will refactor in a different commit to handle if an already made automated_df is passed in ([`261fee7`](https://github.com/UCSD-E4E/PyHa/commit/261fee7d3a637e3e9cc539046bf1059310114142))

* fixed merge conflict ([`59b2579`](https://github.com/UCSD-E4E/PyHa/commit/59b2579aa94f442826a9960e5b27ce8bcafeff8e))

* got the visualizations to work and fixed the local score output to be more reasonable ([`0aa91d3`](https://github.com/UCSD-E4E/PyHa/commit/0aa91d369802a9175167e882e364b6f522d2ab03))

* Update README.md

Updated example to include the &#34;model&#34; isolation parameter ([`be77ddb`](https://github.com/UCSD-E4E/PyHa/commit/be77ddb3aff8d6c0ba42de3c02075c6a46487ec3))

* Documentation update for new BirdNET-Lite functions ([`e6190c6`](https://github.com/UCSD-E4E/PyHa/commit/e6190c6fc9181dd6a4b38cc7ba163075b51e59b7))

* Merge branch &#39;birdnet_integration&#39; of https://github.com/UCSD-E4E/PyHa into birdnet_integration ([`92a31a2`](https://github.com/UCSD-E4E/PyHa/commit/92a31a2f7b0e9ca5adb5aff1115e348aa74d9b9d))

* Fixed typo
birdnet &#34;model&#34; isolation parameter was mispelled as &#34;birndet&#34; ([`d3b8f02`](https://github.com/UCSD-E4E/PyHa/commit/d3b8f020ca063cac5286d0633b7c3241bb73ed46))

* adding environment with pytorch for tweetynet and fixed pyha tutorial for tweetynet ([`fd8e6ff`](https://github.com/UCSD-E4E/PyHa/commit/fd8e6ff0d1687ea6839ca2a68692c9f0ce271020))

* local commit of file ([`ea35605`](https://github.com/UCSD-E4E/PyHa/commit/ea356059aeda26a9b036810567888c806dd8915a))

* adding changes for another file ([`7907803`](https://github.com/UCSD-E4E/PyHa/commit/7907803cde8e9ac8031a16affbb239834a07f281))

* updated files and checkiing results on windows ([`4893339`](https://github.com/UCSD-E4E/PyHa/commit/48933398d8a903ca9d0f7b728ac28a8a10447b52))

* New Ubuntu 20.04 conda environment

- Upgrades to tensorflow 2.8.0 which is compatible with both Microfaune and BirdNET ([`2e8e3e7`](https://github.com/UCSD-E4E/PyHa/commit/2e8e3e7f8551f18e3e412f4fbe58798508a0189b))

* Merge pull request #108 from shreyasar2202/main

Added BirdNet Pipeline ([`ffaec7a`](https://github.com/UCSD-E4E/PyHa/commit/ffaec7ad3aa93b7e58e477706750710387007061))

* Added BirdNet Pipeline ([`e3804e2`](https://github.com/UCSD-E4E/PyHa/commit/e3804e201850f6212edf27dc53c6f96ba66047ea))

* adding tweetynet model to pyha ([`e3fbbba`](https://github.com/UCSD-E4E/PyHa/commit/e3fbbba2a956923a068cb5760c83e553881cbfbd))

* Merge branch &#39;main&#39; of https://github.com/UCSD-E4E/PyHa into main ([`9de4cce`](https://github.com/UCSD-E4E/PyHa/commit/9de4cce8c3595c8f9f9aee7a5a5e929ff808d4ca))

* Added extra error handling to generate_automated_labels
- Caught the case where an empty wav file is successfully loaded in
- Problem was that in these cases the script would crash whenever the signal was downsampled ([`18e5bf7`](https://github.com/UCSD-E4E/PyHa/commit/18e5bf7e3ff1160f71b05b889104c032e6e7a1ef))

* Merge pull request #99 from sprestrelski/kaleidoscope_folder

Fix &#34;FOLDER&#34; column Kaleidoscope compatibility ([`1a4cccd`](https://github.com/UCSD-E4E/PyHa/commit/1a4cccd5d1dbccf46c4cc61a0be724e03523cb13))

* Fix &#34;FOLDER&#34; column Kaleidoscope compatibility ([`d6a1069`](https://github.com/UCSD-E4E/PyHa/commit/d6a1069c599b7429f9a5062d89a83da15c86aa0f))

* Update IsoAutio.py

- Fixed a comment on the resampling portion of the code base. ([`75cca40`](https://github.com/UCSD-E4E/PyHa/commit/75cca4040e5fca696a51ea1be5e7f2615430c35f))

* Merge pull request #98 from UCSD-E4E/Resampling-generalizability

Made resampling more general ([`8f18854`](https://github.com/UCSD-E4E/PyHa/commit/8f18854f398483392fd5f61fac4f5990333a8d92))

* Made resampling more general

Scipy&#39;s resample function allows for both upsampling and downsampling, made sure that this function also includes upsampling, not just downsampling. ([`a8b0c04`](https://github.com/UCSD-E4E/PyHa/commit/a8b0c04e0ae1945c280cb9f4067fcbddb94bb7ba))

* Update README.md (#95) ([`3896104`](https://github.com/UCSD-E4E/PyHa/commit/3896104cf5a9cb8d7f9e2ea102e3f321d048bfeb))

* Merge pull request #94 from UCSD-E4E/readme-real-fix

Fixed broken logo ([`3a94c0f`](https://github.com/UCSD-E4E/PyHa/commit/3a94c0f3accacd88330ad7cb48d627e62a58713d))

* Fixed broken logo ([`52cfef1`](https://github.com/UCSD-E4E/PyHa/commit/52cfef125e842c5c6bbe49ac4279a7889fa3c15e))

* Merge pull request #93 from UCSD-E4E/readme-fix

Fixed missing logo ([`79d015e`](https://github.com/UCSD-E4E/PyHa/commit/79d015e5dbeff9c342f6b525b631a46e4c24e9c4))

* Update README.md ([`9ae743d`](https://github.com/UCSD-E4E/PyHa/commit/9ae743dcd2b2dbe14c13c0b645ac7b57f657a124))

* Merge pull request #92 from UCSD-E4E/staging

Staging ([`bc05e73`](https://github.com/UCSD-E4E/PyHa/commit/bc05e73c25599a5025cf2ed9aea41b8b8c98dd25))

* Updated Tutorial Notebook
Added more relevant demo of the annotation histogram ([`20d0ad8`](https://github.com/UCSD-E4E/PyHa/commit/20d0ad81b9209e34eda6e1f375fa6e34bd8ed506))

* Annotation Length Histograms  (#89)

* move from old branch

* update tutorial

* update

* Add parameters update to match

* update tutorial ([`aecd8f5`](https://github.com/UCSD-E4E/PyHa/commit/aecd8f5b7dfe837b09d7cccdcf312bb6142e51ab))

* Merge pull request #81 from UCSD-E4E/linting

Initial linting with PEP8 ([`a2d7299`](https://github.com/UCSD-E4E/PyHa/commit/a2d729961ad6ad60d14da5dcf27f338f31edfa20))

* Merge branch &#39;staging&#39; into linting ([`9a0fed3`](https://github.com/UCSD-E4E/PyHa/commit/9a0fed31a2cc0db242c8bacd9bc6d054df7d1716))

* Merge pull request #87 from UCSD-E4E/gitignore

Created a .gitignore file ([`da2989b`](https://github.com/UCSD-E4E/PyHa/commit/da2989b277f04787c39f92254c82b4faa62f4e02))

* deleted pycaches ([`365e900`](https://github.com/UCSD-E4E/PyHa/commit/365e90041c992336e1fc6fa605bd455f01be11a0))

* gitignore ([`27cc011`](https://github.com/UCSD-E4E/PyHa/commit/27cc011bdb881fc8897d3731d79b444073d4347c))

* Readme (#86)

* Update README.md

removed last remnant of old name on this branch

* Create artwork

* Add files via upload

* Update ver3 zoo2.svg

* Update ver3 zoo2.svg

* Delete ver3 zoo2.svg

* Delete ver3 zoo3.svg

* Delete ver3 zoo5.svg

* Add files via upload

* Add files via upload

* Add files via upload

* Add files via upload

* Add files via upload

* Add files via upload

* Add files via upload

* Create IsoAutio.svg

* Delete ver3_zoo8.svg

* Delete ver3_zoo5.svg

* Delete ver3_zoo1.svg

* Delete ver3_zoo2.svg

* Delete ver2.svg

* Delete ver1.svg

* Create PyHa

* Rename PyHa to PyHa.svg

* Delete ver3_zoo3.svg

* Delete ver3_zoo7.svg

* Delete artwork

* Created separate folder for conda environments, added Ubuntu 18.04 environment

* Removing old Ubuntu 16.04 environment

No longer needed since the last commit added the conda_environments folder

* Added in a Windows 10 environment file

Tested on one machine

* Add files via upload

Adding environment .yml file for macOS Big Sur 11.4

* isolate function

* Update README.md

* isolation_parameters

* threshold function

* return values

* Update README.md

* Update README.md

* usage

* file links

* Locations

* isolation_parameters update

* isolation_parameters update

* design

* Update README.md

* Update README.md

* steinberg_isolate

* Table of Contents

* PyHa Logo

* logo

* steinberg isolate

* update logo

* steinberg isolate

* isolation techniques

* rename variables

* generate_automated_labels

* generate_automated_labels

* kaleidoscope_conversion

* All IsoAudio.py files

* section headings

* local_line_graph

* local_score_visualization

* plot_bird_label_scores

* All visualizations.py functions

* comments

* microfaune_package notes

* annotation_duration_statistics

* bird_label_scores

* automated_labeling_statistics

* global_dataset_statistics

* clip_IOU

* matrix_IoU_Scores

* clip_catch

* global_IOU_Statistics

* dataset_Catch

* dataset_IoU_Statistics

* All statistics.py functions

* Sections

* sections

* update dataset_IoU_Statistics

* Update image

* Formatting

* Update Image

* Update installation directions

* change to numbered list

* Update Installation Directions

* Update README.md

* Examples, Remove &#34;How it Works&#34;

* Update README.md

* Update README.md

* isolation params in example

* Examples

* Update Graphs

* Update Style

* Credit

* Credit update

* Credits updates

* Update README.md

removed &#34;Install Jupyter Notebook&#34; step
This should be accomplished by the conda environment install.

* Update README.md

Adjusted Examples portion. Results were created and pushed from my Linux Mint laptop which is built off of Ubuntu 16.04

* Update Readme.md with relevant Screaming Piha Test set information

Would be smart to get someone that is new with PyHa to see if they can easily get things up and running simply by reading the readme. I also added in more details to the steps to make them more foolproof, might be overkill.

* Updated Images

Co-authored-by: Jacob &lt;jgayers@ucsd.edu&gt;
Co-authored-by: NathalieFranklin &lt;69173439+NathalieFranklin@users.noreply.github.com&gt;
Co-authored-by: Jacob &lt;jacobthescreenwriter@gmail.com&gt;
Co-authored-by: mugen13lue &lt;33683103+mugen13lue@users.noreply.github.com&gt; ([`796bd94`](https://github.com/UCSD-E4E/PyHa/commit/796bd94d7a8a9881ec3119b052bc8b7a876d3573))

* Revert &#34;Address Linting Warnings&#34;
- Local repo wasn&#39;t properly updated, leading to the undoing of important recent readme PR request

This reverts commit 684a03c9e4af97dd0a3d0e5c30b713969a8e1440. ([`3bb9c8a`](https://github.com/UCSD-E4E/PyHa/commit/3bb9c8a35b421fe2ed256d0b2f72997a1f5e842c))

* Address Linting Warnings
- Related to unused variables
- Fixed IsoAutio import in the visualizations.py file
- master_clips_stats_df ==&gt; master_clip_stats_df ([`684a03c`](https://github.com/UCSD-E4E/PyHa/commit/684a03c9e4af97dd0a3d0e5c30b713969a8e1440))

* removed all whitespace and styling errors ([`0a5a879`](https://github.com/UCSD-E4E/PyHa/commit/0a5a879bba9009d7adc06386a163a7b41c8dbdf1))

* IsoAutio.py update ([`331c1aa`](https://github.com/UCSD-E4E/PyHa/commit/331c1aa23928b0d5f9c1d604a0ef7c80c375f400))

* Reworked audio test set.
- Decided it best to use the Creative Commons xeno-canto data
- Decided it best to stick to Screaming Piha audio to honor the package&#39;s name
- Had to update the Jupyter Notebook tutorial with the new dataset
- Deleted the audio, new update in the readme branch will link to a public drive with the relevant test set
- Replaced the old labels from the old test set with the labels for the new screaming piha dataset. ([`62ba381`](https://github.com/UCSD-E4E/PyHa/commit/62ba3815880765236f4d3dc679120aca393dedcf))

* initial linting with pep8 ([`13fcab1`](https://github.com/UCSD-E4E/PyHa/commit/13fcab1f024983a0aa3afa433d77dd3c29fa2aa3))

* Merge pull request #80 from UCSD-E4E/staging

Demo local_score_visualization() changes ([`e1546bc`](https://github.com/UCSD-E4E/PyHa/commit/e1546bc3725d9f14bf47fa3dd210349048c785b7))

* Demo local_score_visualization() changes ([`c143568`](https://github.com/UCSD-E4E/PyHa/commit/c14356804e15c11003bf2d3f0b22ed0a580e2528))

* Merge pull request #79 from UCSD-E4E/staging

Staging ([`440cab6`](https://github.com/UCSD-E4E/PyHa/commit/440cab643d87524bfd64eeddb39780b9ac28c790))

* Improved local_score_visualizations()
- Allowed a user to insert whatever sort of pre-rendered annotations that they so desire.
- Allow them to change the words on the legend that appear corresponding to the annotations ([`92dfee2`](https://github.com/UCSD-E4E/PyHa/commit/92dfee21699278673b6e5e4900d77b829f31a442))

* Improved visualizations.py error handling
- Handled more potential points of failure
- Added better descriptions of what may have gone wrong ([`237958a`](https://github.com/UCSD-E4E/PyHa/commit/237958a86678ac14dff6784341e00b58215e500f))

* Merge pull request #78 from UCSD-E4E/staging

Staging ([`e82aa02`](https://github.com/UCSD-E4E/PyHa/commit/e82aa02f6505921cd43291ed7f9193fcbb44a92b))

* Fixed PyHa Tutorial Notebook
- Recent commit didn&#39;t display everything ([`530bd33`](https://github.com/UCSD-E4E/PyHa/commit/530bd33d7ee63087f8fa28a13ad6f028a007e7ff))

* Improved visualizations.py error handling
- local_score_visualization() lets you know which clip failed to receive microfaune predictions ([`663f3be`](https://github.com/UCSD-E4E/PyHa/commit/663f3bec1c0f354676e53233813942dac71d3aab))

* Merge pull request #77 from UCSD-E4E/staging

Staging ([`10a5ff9`](https://github.com/UCSD-E4E/PyHa/commit/10a5ff9daaa764926c6f32c158bb8f86bf5b73ca))

* Improved zero division error-handling messaging
- Specific to the matrix_IoU_Scores() function ([`a3e25b7`](https://github.com/UCSD-E4E/PyHa/commit/a3e25b7ef5fc55b26beaedabfda722de22303d71))

* Changing bird_dir to audio_dir in IsoAutio.py
- This is part of the transition to make PyHa more general. ([`740b51c`](https://github.com/UCSD-E4E/PyHa/commit/740b51c3c12af03510ed7dde1afc5c345c489cc7))

* Converted bi_directional_jump to window_size
- This is a more standard description of what the steinberg isolation technique is deploying ([`f5350de`](https://github.com/UCSD-E4E/PyHa/commit/f5350dede8960c8f2e130e1017d8b72767281b91))

* Merge pull request #75 from UCSD-E4E/staging

Staging ([`95a06cb`](https://github.com/UCSD-E4E/PyHa/commit/95a06cb9b62de94289112e77ddc2c914245f1a7c))

* Updated PyHa_Tutorial Notebook with new labels ([`ab2b567`](https://github.com/UCSD-E4E/PyHa/commit/ab2b56718d33f3ffca13568a870199827b075c8f))

* Added columns for Kaleidoscope compatibility
- Channel had to be added. LABEL was changed to MANUAL ID ([`29b3d9b`](https://github.com/UCSD-E4E/PyHa/commit/29b3d9b449d4beb12391173e522f04a86e6ce7bb))

* Merge pull request #74 from UCSD-E4E/staging

Improved Manual Labels for test set ([`3d84cbf`](https://github.com/UCSD-E4E/PyHa/commit/3d84cbf4f21dd51d12e0a5baceef8af0be321b3b))

* Improved Manual Labels for test set
- Annotations re-done by Jacob using Pyrenote ([`2963a23`](https://github.com/UCSD-E4E/PyHa/commit/2963a23e27ec1180791f8dde72c1614ad9eeefdd))

* Merge pull request #69 from UCSD-E4E/staging

Staging ([`29aabca`](https://github.com/UCSD-E4E/PyHa/commit/29aabca44f5e785ca80389af406a5e31d209badd))

* Forgot to add continue to previous commit ([`e071979`](https://github.com/UCSD-E4E/PyHa/commit/e07197934651c082796400a13b2fdefce0f97312))

* Added try-except block to handle faulty wav files
- Found in situations where you want to run an isolation algorithm across a large set of wave files that haven&#39;t been properly vetted for various problems that can occur. Such as RIFX instead of RIFF or wave files that were created but failed to actually record anything and are empty.
- I am not that experienced with error handling, but this change made it work on a large folder filled with Audiomoth clips that had tons of errors ([`fcf6729`](https://github.com/UCSD-E4E/PyHa/commit/fcf67296a11dc18d86d83f2f019942f86412b21c))

* Merge pull request #68 from UCSD-E4E/Environment

Added environment File for MacOS ([`dfdc528`](https://github.com/UCSD-E4E/PyHa/commit/dfdc528625684197288f91a099880878bbad5055))

* Add files via upload

Adding environment .yml file for macOS Big Sur 11.4 ([`e6c91c3`](https://github.com/UCSD-E4E/PyHa/commit/e6c91c3de3d411bab3cd56b9938dc801ccb3b7f4))

* Added in a Windows 10 environment file

Tested on one machine ([`673fd69`](https://github.com/UCSD-E4E/PyHa/commit/673fd694703361fbf8caddbdca340bac672c5f41))

* Removing old Ubuntu 16.04 environment

No longer needed since the last commit added the conda_environments folder ([`e6caba5`](https://github.com/UCSD-E4E/PyHa/commit/e6caba5109d7a5cc089ca8553a9eee85d949f95f))

* Created separate folder for conda environments, added Ubuntu 18.04 environment ([`6ab6f60`](https://github.com/UCSD-E4E/PyHa/commit/6ab6f60f354e2ad6b0c93f5df6c98f9b2a723acb))

* Merge pull request #67 from UCSD-E4E/art

New art for read me ([`bee421e`](https://github.com/UCSD-E4E/PyHa/commit/bee421e215116328667827c96c393530fe04ca0d))

* Delete artwork ([`723f7cc`](https://github.com/UCSD-E4E/PyHa/commit/723f7cc52efd017bdcb91ee613396375986f8884))

* Delete ver3_zoo7.svg ([`6f87b02`](https://github.com/UCSD-E4E/PyHa/commit/6f87b029bf21b64f526a2065e3daf5fedbd9bda5))

* Delete ver3_zoo3.svg ([`ba4d39b`](https://github.com/UCSD-E4E/PyHa/commit/ba4d39bc3226e41f9041599672a2d2491e01981c))

* Rename PyHa to PyHa.svg ([`10aae25`](https://github.com/UCSD-E4E/PyHa/commit/10aae25eb9b01ba80d8a6b5115551144ddf72294))

* Create PyHa ([`181222a`](https://github.com/UCSD-E4E/PyHa/commit/181222a8188c5a51ed8359032bef2deb694c0523))

* Delete ver1.svg ([`54346f1`](https://github.com/UCSD-E4E/PyHa/commit/54346f110786ab2701898d81d92daf421e6bad48))

* Delete ver2.svg ([`48d07ce`](https://github.com/UCSD-E4E/PyHa/commit/48d07ce67ad2807abf8d3f71657046d5492abc13))

* Delete ver3_zoo2.svg ([`ff8f93b`](https://github.com/UCSD-E4E/PyHa/commit/ff8f93b9f3746b75b59627339c61796ab88664b3))

* Delete ver3_zoo1.svg ([`9e8553e`](https://github.com/UCSD-E4E/PyHa/commit/9e8553ea7bebbe96bde918c6d343b53662adae90))

* Delete ver3_zoo5.svg ([`06712f6`](https://github.com/UCSD-E4E/PyHa/commit/06712f61e367aadd2e7c1290c9462128feb9b662))

* Delete ver3_zoo8.svg ([`3d1ef82`](https://github.com/UCSD-E4E/PyHa/commit/3d1ef827c5747fe54c9ac564a0f26880c370b570))

* Create IsoAutio.svg ([`5aefe43`](https://github.com/UCSD-E4E/PyHa/commit/5aefe4356438284ed45bbdaf2a43b3a07ab7857c))

* Merge pull request #65 from UCSD-E4E/staging

Staging ([`c018e4f`](https://github.com/UCSD-E4E/PyHa/commit/c018e4f3f7243e6e2857e551cf798ca6d6d651dc))

* Updated PyHa tutorial with statistics changes ([`07d2668`](https://github.com/UCSD-E4E/PyHa/commit/07d266855db3589381e4c8462c29f7023727aac3))

* IoU code added to automated_labeling_statistics
- Still need to make code general to multiple classes, might rename automated_labeling statistics to class_automated_labeling_statistics
- Some functionality needs to be included in this function, but I want to make sure I have a strong game plan of how to include said features
- These include allowing for two outputs, such as having the global statistics imbued in this function. This could also include having the dataset_IoU functionality where all of the IoU scores are attached to the manual dataframe
- For right now this is a good step in the direction of streamlining the code base. ([`2946356`](https://github.com/UCSD-E4E/PyHa/commit/294635697ee0d9dfcf3d25082fbffae8daf73c95))

* Add files via upload ([`4f8e778`](https://github.com/UCSD-E4E/PyHa/commit/4f8e778e172375116b8e0bc88428177f26c86570))

* Add files via upload ([`82ecf3b`](https://github.com/UCSD-E4E/PyHa/commit/82ecf3be0a9be6b61ace4c96a69af10391b5ea31))

* Add files via upload ([`284690b`](https://github.com/UCSD-E4E/PyHa/commit/284690bceb97fd36448c7aec7679134294fe2e57))

* Add files via upload ([`b489d71`](https://github.com/UCSD-E4E/PyHa/commit/b489d7162d545951f49e340879ee1504c6200b84))

* Add files via upload ([`76b58a7`](https://github.com/UCSD-E4E/PyHa/commit/76b58a79d3888e83f71ad2ae5f2504c7843e8af9))

* Add files via upload ([`bbf63bd`](https://github.com/UCSD-E4E/PyHa/commit/bbf63bdfc4681abcb0005d763723ad8d8c36a55d))

* Add files via upload ([`946b9fa`](https://github.com/UCSD-E4E/PyHa/commit/946b9fa3d887aa7fae999701d71683c9f6ab090a))

* Delete ver3 zoo5.svg ([`33a3114`](https://github.com/UCSD-E4E/PyHa/commit/33a31146e62a3ad32bb51679b138736a07951bb7))

* Delete ver3 zoo3.svg ([`235c584`](https://github.com/UCSD-E4E/PyHa/commit/235c584201aec28f911f4e069f3a93f98d0d8f40))

* Delete ver3 zoo2.svg ([`c29d036`](https://github.com/UCSD-E4E/PyHa/commit/c29d0366663cb1bace84d5e6a3b3eaae7fb03419))

* Update ver3 zoo2.svg ([`c57ae74`](https://github.com/UCSD-E4E/PyHa/commit/c57ae74ce8be42ef81bc02c4aa4acf7c2d60b8c7))

* Update ver3 zoo2.svg ([`dbfa2ea`](https://github.com/UCSD-E4E/PyHa/commit/dbfa2eaf6aaa63c062ec26c6490b2751db6334c4))

* Add files via upload ([`9bc055c`](https://github.com/UCSD-E4E/PyHa/commit/9bc055c9151648443897dc4b6213387baf9485f2))

* Create artwork ([`7d9b0d5`](https://github.com/UCSD-E4E/PyHa/commit/7d9b0d502413c8e43e000fe1cc6e0eaaf5b3743b))

* Update README.md

removed last remnant of old name on this branch ([`b5a2525`](https://github.com/UCSD-E4E/PyHa/commit/b5a252523cb6f70f6a1fc1241ea65301e15699e8))

* Merge pull request #60 from UCSD-E4E/staging

Staging ([`b50ea74`](https://github.com/UCSD-E4E/PyHa/commit/b50ea7455d87da5a7b4aa8ba3236fef91b95234b))

* Merge branch &#39;staging&#39; of https://github.com/UCSD-E4E/PyHa into staging ([`d05d5f8`](https://github.com/UCSD-E4E/PyHa/commit/d05d5f8048b1060f713852b3397aee6c2f61197e))

* Fleshed out documentation on statistics.py
- Added in first pass of documentation to IoU statistics related functions
- Added more descriptions that are with respect to how the project has evolved in the recent months. ([`017a336`](https://github.com/UCSD-E4E/PyHa/commit/017a336e934d510e13be0b5055cf7d44a84fdd44))

* Merge pull request #59 from UCSD-E4E/main

Synchronizing Master and Main ([`16affb4`](https://github.com/UCSD-E4E/PyHa/commit/16affb4d9d1b2e0d8df2b80354d16a60048d92ee))

* Fleshed out Documentation of IsoAutio functions
- Pseudocode has been added to all 4 current isolation techniques
- Adjusted all function docstrings for their new parameters
- Added on some clarifications of function summaries ([`1474406`](https://github.com/UCSD-E4E/PyHa/commit/14744065f37eeb123a5f5efa9c011749d00419da))

* Adding threshold min to tutorial ([`33ebdc3`](https://github.com/UCSD-E4E/PyHa/commit/33ebdc3b9c48c52285dd5a6947456136d3a49f26))

* Merge pull request #56 from UCSD-E4E/master

Master ([`e64e5d6`](https://github.com/UCSD-E4E/PyHa/commit/e64e5d6688560ab1e5ced1bd2a2dfa748eb6bba7))

* Delete IsoAutio_Tutorial.ipynb

replaced by PyHa_Tutorial.ipynb ([`99406a5`](https://github.com/UCSD-E4E/PyHa/commit/99406a5376b1c2d9533e9369ac0f3354df73cb88))

* Delete IsoAutio directory

Replaced by the PyHa folder ([`ef64882`](https://github.com/UCSD-E4E/PyHa/commit/ef64882ca23d024738c014e1d36f888a3cb70728))

* Changes to IsoAutio Local Score Thresholds
- Created a new threshold() function that defines the isolation threshold based on the local score array and the isolation parameters
- Added in a new key to isolation parameters called threshold_min. This helps reduce the number of false positives accumulated from using relative thresholds to lump local scores together.
- Ensured that threshold min has been integrated into all four isolation techniques. ([`89a3b57`](https://github.com/UCSD-E4E/PyHa/commit/89a3b57dd931738940ebf942de251d0adb3af9ed))

* Merge pull request #55 from UCSD-E4E/master

Master ([`6281cc3`](https://github.com/UCSD-E4E/PyHa/commit/6281cc3d18fd2a65b9bf4b6c15d323002d12761a))

* Delete IsoAutio directory

- Transitioning from IsoAutio as the working title to PyHa as the name of the Python package. 
- isolation.py has already been renamed to IsoAutio.py
- All of the logic in this directory is now in the PyHa directory. Plus I made a couple of changes directly to PyHa before deleting this. ([`bd900b3`](https://github.com/UCSD-E4E/PyHa/commit/bd900b3fff8246fb15bfe1f55d44b261fc7dbe9c))

* Delete IsoAutio_Tutorial.ipynb

Replaced with PyHa_Tutorial.ipynb ([`faa708e`](https://github.com/UCSD-E4E/PyHa/commit/faa708e6c5a95d67be631bda06e449980565d263))

* Fixed PyHa_Tutorial Jupyter Notebook
- The outputs weren&#39;t compiled ([`0fe1d2c`](https://github.com/UCSD-E4E/PyHa/commit/0fe1d2c71b3560be66678d66c046dd85fec4e1c2))

* Adjusting some folder and filenames for &#34;PyHa&#34;
- Had to modify some files import statements
- added some new comments to some functions that I looked back over.
- Adjusted the jupyter notebook tutorial ([`f81bf76`](https://github.com/UCSD-E4E/PyHa/commit/f81bf76fa6d37023961a79f82b0c71ed91db7338))

* update Readme to account for renaming the package to PyHa ([`6d31f86`](https://github.com/UCSD-E4E/PyHa/commit/6d31f86ce30b754ccad5249577a8193d8407aa10))

* Merge pull request #53 from UCSD-E4E/master

Removed plot_bird_label_scores() from statistics.py ([`1b67471`](https://github.com/UCSD-E4E/PyHa/commit/1b67471bcfb688bc0fc713247ba20d9485ea02a8))

* Removed plot_bird_label_scores() from statistics.py
- Forgot to remove it when I migrated all of the matplotlib related functions into visualizations.py ([`6c7b04c`](https://github.com/UCSD-E4E/PyHa/commit/6c7b04c177a8c057ed5e05ed22a60ac905c19dea))

* Merge pull request #52 from UCSD-E4E/master

Added Kaleidoscope converting function ([`eec53f9`](https://github.com/UCSD-E4E/PyHa/commit/eec53f9d4014412ebf308952ebc6491b9f974f17))

* Added Kaleidoscope converting function
- Function strips away columns in the manual or automated dataframes that we need for our package that would otherewise be incompatible with Kaleidoscope. ([`cda7e6f`](https://github.com/UCSD-E4E/PyHa/commit/cda7e6f8622f1ddc2ff1d56e57f22680e6adf421))

* Merge pull request #46 from UCSD-E4E/master

Master ([`47a1ae3`](https://github.com/UCSD-E4E/PyHa/commit/47a1ae317c061db3126e15fc8794fec92c299dc3))

* Fixed microfaune package import
- Wasn&#39;t addressing microfaune_package properly relative to visualizations.py and isolation.py ([`31e58a5`](https://github.com/UCSD-E4E/PyHa/commit/31e58a55500267a19283bee76db51c63b18a073a))

* Reorganized microfaune_local_score.py
- Separated it into visualizations.py, statistics.py, and isolation.py all in the IsoAutio folder
- Updated Microfaune_Local_Score_Tutorial jupyter notebook with the new changes.
- Looking to rebrand Automated_Audio_Labeling_System_AID to IsoAutio (going to double-check that everyone is happy with this before changing the repo name)
- separated bird_label_scores() into two separate functions bird_label_scores() and plot_bird_label_scores() so that they fit cleanly into statistics.py and visualizations.py respectively. ([`0ded6e9`](https://github.com/UCSD-E4E/PyHa/commit/0ded6e993450472f8732f189511501446225aa45))

* Merge pull request #44 from UCSD-E4E/master

Removed True Negative from IoU Related Statistics ([`4191d37`](https://github.com/UCSD-E4E/PyHa/commit/4191d37623280da8306ba8ed05b22c654cb9ce77))

* Removed True Negative from IoU Related Statistics
- Turns out that True Negatives are irrelevant to Object Detection based problems ([`4909df1`](https://github.com/UCSD-E4E/PyHa/commit/4909df1049bce240cb73f0ee4bc824bc3a54c909))

* Merge pull request #40 from UCSD-E4E/master

Added in new &#34;chunk&#34; isolation technique ([`163dec4`](https://github.com/UCSD-E4E/PyHa/commit/163dec4c9f87e609230f98a9b07ff0323e3eeeb8))

* Added in new &#34;chunk&#34; isolation technique
- New technique should be handy in situations where audio annotations are a discrete length. Cases such as BirdNET outputs as well as the BirdCLEF2020 labels.
- Also handled a bug related to IoU scores in bird_label_score() ([`8a03ffd`](https://github.com/UCSD-E4E/PyHa/commit/8a03ffd6bc32912faba2c0293c17789a726d626c))

* Merge pull request #36 from UCSD-E4E/master

Fixed standard deviation calculation errors and allowed normalization of local score arrays ([`d464692`](https://github.com/UCSD-E4E/PyHa/commit/d464692dd56a5d301c69028c5ec5c58ef0452904))

* Added in the ability to normalize the local scores.
- This came up due to the fact that the relative scores can be fairly small in our work, and this will make it easier to perform pure thresholds.
- There are other tweaks included in this commit ([`9764300`](https://github.com/UCSD-E4E/PyHa/commit/97643009041e1d04e4750b7bd663b86957fb9501))

* Fixed error in standard deviation threshold type
- I was deriving the threshold purely from standard deviation values rather than the mean + (std_dev + threshold_const) ([`a2ea9dd`](https://github.com/UCSD-E4E/PyHa/commit/a2ea9dd1364a0da784d0f244e8cc47ec270cb886))

* Merge pull request #35 from UCSD-E4E/main

Updating Master ([`5d5f800`](https://github.com/UCSD-E4E/PyHa/commit/5d5f800ea9b135ae1303154fe648d0ef5854d343))

* Merge pull request #34 from UCSD-E4E/JacobGlennAyers-patch-1

Jacob glenn ayers patch 1 ([`1a853a1`](https://github.com/UCSD-E4E/PyHa/commit/1a853a1478638b9f9193d2de9896c9cbf5771392))

* Merge pull request #33 from UCSD-E4E/master

Master ([`6e41cb5`](https://github.com/UCSD-E4E/PyHa/commit/6e41cb58459ee8d9385ecefbccd632079fdcff95))

* Adding in fixed Jupyter Notebook Tutorial

Not sure why the Notebook didn&#39;t run all of the cells on the last push ([`676e701`](https://github.com/UCSD-E4E/PyHa/commit/676e701d6289ec9b0bce4ae174952641d312f041))

* Delete Microfaune_Local_Score_Package_Tutorial.ipynb ([`3420806`](https://github.com/UCSD-E4E/PyHa/commit/3420806dcbed434ecda9cf0ac034f51252d772ed))

* Fixed Jupyter Notebook Tutorial ([`5e39378`](https://github.com/UCSD-E4E/PyHa/commit/5e3937887af46f6a8b1fc90a35751bdf72b25d68))

* Merge branch &#39;master&#39; of https://github.com/UCSD-E4E/Automated_Audio_Labelling_System_AID.git; branch &#39;master&#39; of https://github.com/UCSD-E4E/Automated_Audio_Labelling_System_AID
- working to fix the jupyter notebook tutorial ([`8055db6`](https://github.com/UCSD-E4E/PyHa/commit/8055db6eb935ff59226c100ea1201e447c3f4df5))

* Merge branch &#39;master&#39; of https://github.com/UCSD-E4E/Automated_Audio_Labelling_System_AID ([`81f43ce`](https://github.com/UCSD-E4E/PyHa/commit/81f43cee5f55b818ffcce28c93ee6b6179792ffa))

* Merge branch &#39;main&#39; into master ([`104e911`](https://github.com/UCSD-E4E/PyHa/commit/104e911dc517452a852675b3cfa7859bb8a4105d))

* Added in new &#34;stack&#34; isolation technique
- Tested to make sure that it integrates into all of the other functions
- Added True Negative values to IoU statistics, updated jupyter notebook accordingly. ([`26f4296`](https://github.com/UCSD-E4E/PyHa/commit/26f429605d388c2905c03ff12ff6226f08a91f5a))

* Merge pull request #28 from UCSD-E4E/master

Enabled user tweaking of isolation parameters ([`8150873`](https://github.com/UCSD-E4E/PyHa/commit/81508734e91eafea2d956f5ea7e70694f5dc81e5))

* Merge branch &#39;main&#39; into master ([`d87634c`](https://github.com/UCSD-E4E/PyHa/commit/d87634c25359b8175f891e2601a6d137332c6118))

* Enabled user tweaking of isolation parameters
- Encapsulated the &#34;steinberg&#34; and &#34;simple&#34; techniques into a master &#34;isolate&#34; function that is defined by the passed in isolation parameters
- Renamed calc_local_scores() to generate_automated_labels()
- Reworked generate_automated_labels() to take in the isolation parameters dictionary
- Reworked local_score_visualization() to take in the isolation parameters as well as the master isolate() function.
- Deleted the calc_local_scores2() and local_score_visualization2() functions that were designed to demonstrate the simple isolation technique.
- updated jupyter notebook tutorial to demonstrate above changes.
- removed the second howler monkey example clip visualizations as well as the demonstration of &#34;catch&#34; scores since they seem redundant and inflated the size of the tutorial ([`6d4d11b`](https://github.com/UCSD-E4E/PyHa/commit/6d4d11bfa69c83890ddb0820cbb247bbe38fa47b))

* Merge pull request #27 from UCSD-E4E/revise-documentation

Add docstring comments to functions ([`12887b6`](https://github.com/UCSD-E4E/PyHa/commit/12887b6503f09c8c7febaaaee748dd29e0a09a1c))

* merge main ([`ff6296b`](https://github.com/UCSD-E4E/PyHa/commit/ff6296b4d6b28c89801c3a5398745d0bf9b9fa2c))

* Merge pull request #25 from UCSD-E4E/master

Statistics Function for annotation lengths ([`1b402a9`](https://github.com/UCSD-E4E/PyHa/commit/1b402a99dc7248fc311bf501467c4a1b8b1e409f))

* Statistics Function for annotation lengths
Added in a new function annotation_duration_statistics() which outputs the number of annotations, mode, mean, standard deviation, min, Q1, Median, Q3, and max values with respect to the lengths of the annotations ([`a2ce49c`](https://github.com/UCSD-E4E/PyHa/commit/a2ce49c9f6e8be82d36f1eb2857dc9ca67b326cb))

* update documentation with arg types ([`4ebea63`](https://github.com/UCSD-E4E/PyHa/commit/4ebea6307a8e27300f48bde241d6a34e0f38d2f8))

* merge commit ([`6fe57a0`](https://github.com/UCSD-E4E/PyHa/commit/6fe57a0d6dee7b2e0210f363da2c53dfd5d2d14e))

* added docstring documentation ([`efe3462`](https://github.com/UCSD-E4E/PyHa/commit/efe34626a1ba63b6d546a12ec46f6a7718803e51))

* Merge pull request #24 from UCSD-E4E/master

Added log scale functionality to visualizations ([`9c5d4ef`](https://github.com/UCSD-E4E/PyHa/commit/9c5d4ef9c3ead7a96fc5b82366b2b1d18de50c0a))

* Added log scale functionality to visualizations
This makes it easier on human eyes. Have local_score_visualization() defaulted to a linear scale fixed between [0,1] ([`0d06b02`](https://github.com/UCSD-E4E/PyHa/commit/0d06b02dd775038fca748af115f84c38d52befe3))

* Merge pull request #21 from UCSD-E4E/master

Quick fix to Jupyter Tutorial ([`8aa7595`](https://github.com/UCSD-E4E/PyHa/commit/8aa759545296a9724e783b2b19b8847db63251f1))

* Quick fix to Jupyter Tutorial
Made sure new changes with new isolation technique show up compiled in the tutorial ([`939159c`](https://github.com/UCSD-E4E/PyHa/commit/939159c253d43011e0b30322f03c8be800714b44))

* Merge pull request #20 from UCSD-E4E/master

Developed Simpler Approach to Isolating Calls ([`d21c682`](https://github.com/UCSD-E4E/PyHa/commit/d21c682712f6de2ebc0d35c0071a6a5b66a41e32))

* Developed Simpler Approach to Isolating Calls
Any local score &gt;= some threshold is automatically set to a bird call. Will continue as long as there are multiple scores in a row above the threshold. To display, required me to create two new functions that server the purpose of visualization as well as calculating the scores over a folder of wav files. This is a horrendous solution creating two new functions, so I have to go back and rework calc_local_scores() and local_score_visualization() to support different types of isolation. This is something that should be done when I rework Isolate for gradient descent. ([`cadbb09`](https://github.com/UCSD-E4E/PyHa/commit/cadbb097a057464c51a3f8e8b50bda5d05ab60ba))

* Merge pull request #19 from UCSD-E4E/master

Added Recall and F1 scores to IoU Statistics ([`aced641`](https://github.com/UCSD-E4E/PyHa/commit/aced6414e9d34689bc2845fbddecd3c2bd111078))

* Added Recall and F1 scores to IoU Statistics
This involved calculating the number of false negatives in the IoU Matrix scores functions ([`4d55bab`](https://github.com/UCSD-E4E/PyHa/commit/4d55babc4b130d4466cddb081161d550397dca28))

* Merge pull request #18 from UCSD-E4E/master

Added comments and improved readability of isolate ([`712d5be`](https://github.com/UCSD-E4E/PyHa/commit/712d5be296872caab9b78c5be951614a23d9df12))

* Added comments and improved readability of isolate
Laid out the plan for changes to be made to isolate function in order to perform gradient descent. ([`f916a0c`](https://github.com/UCSD-E4E/PyHa/commit/f916a0c6b6963854f4bba7b79e7670414b0166be))

* Merge pull request #16 from UCSD-E4E/master

Added Precision Statistics for IoU Scores ([`bc9d80e`](https://github.com/UCSD-E4E/PyHa/commit/bc9d80e7583daea8bc1ee716b8c6d149117785a7))

* Added Precision Statistics for IoU Scores
This involves a couple new functions that have been updated on the jupyter notebook tutorial. I also added a bunch of comments to microfaune_local_score.py ([`06d9658`](https://github.com/UCSD-E4E/PyHa/commit/06d9658b06575c954f200b0c765e97d633fd7398))

* Merge pull request #14 from UCSD-E4E/master

Added label-by-label IoU and Catch Scores ([`ea030d0`](https://github.com/UCSD-E4E/PyHa/commit/ea030d0c06a3e7a39317ba07a813315dcdeed581))

* Included clip-by-clip IoU and Catch Scores ([`294e079`](https://github.com/UCSD-E4E/PyHa/commit/294e079f90f8026c44d04e62a32a313777a4c929))

* Added conda environment.yaml file ([`1a710bb`](https://github.com/UCSD-E4E/PyHa/commit/1a710bbfb67653254b6aeb084ff03326e7d3bacd))

* Included Conda Environment Information ([`3edcabb`](https://github.com/UCSD-E4E/PyHa/commit/3edcabb9dbac58791aa7bec34311e70a9c4b01e4))

* Delete ScreamingPiha2_Manual_Labels.csv ([`2827d38`](https://github.com/UCSD-E4E/PyHa/commit/2827d38041c8afee5935979e864cc34b4f593370))

* Added IoU Scores and Microfaune Package ([`510fd60`](https://github.com/UCSD-E4E/PyHa/commit/510fd6054b9002921b9e5e94cade382d0df0beab))

* Extra Files for Tutorial.

Does not include the full set of 6 test clips, only brought in 1. ([`198d696`](https://github.com/UCSD-E4E/PyHa/commit/198d696771e2c7a4b87aff418128501db42b4467))

* Bringing over code from passive-acoustic-biodiversity

Contains much of the proof-of-concept work involved in envisioning the potential of this repo. Further work will be done from this repo going forward. ([`321b602`](https://github.com/UCSD-E4E/PyHa/commit/321b6020b6c9cf1787796670bba17a0897f52634))

* Initial commit ([`58111c9`](https://github.com/UCSD-E4E/PyHa/commit/58111c9fd4de2ceea12ea393a31c3030832ebfdd))
