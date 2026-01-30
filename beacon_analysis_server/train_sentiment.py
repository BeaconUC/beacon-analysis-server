from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from beacon_analysis_server.config import MODELS_DIR

negative: set[str] = set(
    [
        "Sus why again?! Kahapon kami ulit",
        "Brown out - 12344556778 Digos - 0 Awan mon Beneco",
        "Wohoooooo! Panalo na ang Feeder 5",
        "Nawala na naman. Ilang beses na ba?",
        "Kala k ba sa 30 pa bakit nawalan ng koryente mga 6 am uala ng koryente",
        "Camp 7?bakt damay dito cabinet hill!??",
        "Fairview hoy",
        "Hala bat po pati central Fairview?6:11am po out of power",
        "Huhu ndi ba pwedeng sinabay nalang sa 5-5 nung last power interruption",
        "thank you BENECO naka duty pa naman ako tozzzz nakasaksak yung laptop ko nung nag on off ng 4x yung kurintiii ba HAHAHAHAHA",
        "Grabe 4hrs na po yang unscheduled. Parang Scheduled na din gnwa niyo e",
        "Sabi sainyo eh bahay lng maganda sa feeder pero yung piyesa BASURA yan matagal ng gawain ng mga electric company ang ganyan kht sino pang pumalit jan same old sh*t Monthly/yearly same old story panay recycle kasi ang piyesa",
        "Grabe beneco may online quiz kami today",
        "Kaka stress naman kaaga aga",
        "Jusko kaninong jumper ba kasi yan?",
        "Unscheduled kagabi around 6pm and again unscheduled today , at iba pa uli yung scheduled power inturruption, really BENECO",
        "Bakit po brown out dito sa t. Alonzo health center e wala naman kayong advisory na mag brown out ngaun?",
        "Parang may sumabog po dito sa Aurora hill,lakas Ng tunog biglang nagbrown out",
        "bakit patI NEW LUCBAN walang kuryente nanaman pati kagabi",
        "Ang gagaling nyo naman. Kahapon brown out ngayon ulit?",
        "Walang kupas ang beneco. Araw araw ang unschedule brown out nila….dapat talaga masa pribado ang buong beneco para sa magandang serbisyo. Kung ang nawawalang kilowat-hour sa mga unschedule brown out ay ma translate sa income palagay ko mas mababa sana ang magiging singil ng beneco. Pero syempre wala silang paki… masisingil naman sa mga paying customer, kaya ok lng.",
        "KANINANG UMAGA PA KAYO UMAY SAINYO SIRA NA MGA APPLIANCES NAMIN SAINYO TAPOS KUNG MAGREREKLAMO MGA TAO SAINYO ANONG SAGOT NINYO EDI WAG KAYONG MAG YAD PARA PUTULIN TANG INANG RESPONSE YAN .. KAYA LUMALAKI BILL S KAGAGUHAN NINYI",
        "kawawa talaga mga naka work from home sainyo, umay..",
        "Unsched poweroutage manen. Haynako palalo nga talaga. richest city kanooooooo",
        "Isang off lng sana wag po ung patay sindi . Naku po mamahal ng appliances mga sir kwawa nmn",
        "Reason: Nilamig yung transformer ng kuryente kaya pumutuk",
        "ANO EVERYWEEK NA LANG?",
        "Nawalan n kaninang umaga! Tas ngayong gabi uli!! Beneco - Benguet Electric Cooperative ano na? Puro band aid solution na lang b talaga kayo??? 2026 na magbago na kayo!!",
        "kung kelan magpapainit ng paligo after 3 days no ligo e. Sige nextweek na.",
        "Grabe kung kailan magiging mabuting studyante na",
        "Kung kailan magluluto na",
        "Bugallon na naman.. lagi na lang may sumsabog jan… kung kelan gabi na..",
        "Naihian nanaman ng aso poste beneco amffff",
        "Kung kelan may online interview (kala ko mapapa finally na magkakawork)hina pa man din ng data lang",
        "Grabe kung kelan kauuwi lang from work at magluluto na nang dinner. Dapat pala kumain na lang ako somewhere bago umuwi.",
        "dapat tangalan kyo ng franchise eh",
        "may engineer pa lyo laging brown out namin",
        "Ano oras nyo po ibabalik ang kutyente dito sa bakakeng every other day na po ganto",
        "Puro unscheduled na lang dito sa kitma. Wala na bang mas maayos na serbisyo?",
        "hindi nyo talaga kaya sundin yung binigay nyo na time na babalik yung kuryente noh?",
        "taena, kawawa appliances sa ginagawa niyo",
        "Mga sinugaling kayo kawawa tao sa inyo 5pm last week ganun din 5pm sabay 6 pala ngayon 730 ukinana",
        "Boss? Anu na? Alas singko na?",
        "wala pa rin pong current sabi til 5pm lang haist on time mag cut ng current tapos laging late magbalik ng kuryente",
        "Until what time po kaya ito? Masisira din po mga gamit kaka 110 ng kuryente.",
        "May work ngani",
        "Almost 1hr na po, not 6:43",
        "Walang abiso nag be bake pa naman ako",
        "2hrs should be enough for a check up from an award winning company",
        "Akala ko nasira bigla electric kettle, ayaw mag ON. Yun pala...natalikod lang ako sandali...",
        "Di pa tapos ang January naka ilang scheduled and unschedule power interruption na kayo. Baka substandard mga gamit niyo kaya palaging sira or sinaaadya nalang to kasi consumer naman nag babayad eh.",
        "Jusko po! Dpa umiinit ang panligo",
        "Almost 2 hours walang update?",
        "pti d2 military cut-off brownout din. Haaayyyzzzz",
        "One hour announcement. paano na lang ang mga may online class huhuhu. buti pala nacheck ko agad",
    ]
)

neutral: set[str] = set(
    [
        "Brownout Digos Meter 123456",
        "Walang kuryente sa Poblacion",
        "No power Fairview",
        "wala pading kuryente dito sa part ng military cut off.",
        "Buti naman binalik agad hahahaha mahal ng singil niyo sa kuryente pero Panay brown out nye.",
        "May sumabog dito Banda sa San Antonio Aurora Hill",
        "No current new lucban",
        "Bayan Park West no power",
        "Walang kuryente dito sa modern site",
        "Any updates?",
        "May ilaw pero 110..victoria village",
        "Wala pong kuryente here sa pinsao proper po.",
        "May pumutok somewhere sa Brookside then nawala na kuryente.",
        "May pumutok na transformer sa may bugallon st.",
        "Kala ko naputulan na kami ng kuryente.",
        "please include in your inspection... asin road km3-4 low voltage.thank you",
        "Update po . What time babalik Ang kuryente",
        "Wala po kami power supply dito sa Chapis Village Bakakeng Central.",
        "Anong oras po magiging 220??",
        "Bat po dto sa kabila ng guitley lubas wala pa po ilaw mam/sir?",
        "BENECO. Please note that there was also reduced voltage in the Lower Bato, Puguis area around that same time. That will destroy people's appliances. You need to focus on making your power supply more reliable and stable instead of installing smart meters. Learn to walk properly before you try to run. Smart meters are notorious for being problematic in other countries.",
        "Anong oras po marestore?",
        "Wala po Quirino Hill",
        "Until what time?",
        "Blackout Baguio Country Club Village, exactly at 4:40pm today, January 22, 2026. Please check.",
        "Any updates din po kaya for us?",
        "Biglaan dito sa camp 7",
        "Wala din Dito SA caponga nmn Ngayon 1/19/26 bakit",
    ]
)

positive: set[str] = set(
    [
        "Salamat! Naayos na ang brownout!",
        "Ambiong po meron na, salamat!",
        "May ilaw na po at 7:38 am...slamat po sir/madam..",
    ]
)

train_data = (
    [{"text": s, "label": 0} for s in negative]
    + [{"text": s, "label": 2} for s in neutral]
    + [{"text": s, "label": 1} for s in positive]
)

# print(len(negative))
# print(len(neutral))
# print(len(positive))
# print(len(train_data))

model_name = "dost-asti/RoBERTa-tl-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)


dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir=f"{MODELS_DIR}/roberta-sentiment-custom",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=5,
    save_steps=50,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
trainer.save_model(f"{MODELS_DIR}/roberta-sentiment-custom")
tokenizer.save_pretrained(f"{MODELS_DIR}/roberta-sentiment-custom")
