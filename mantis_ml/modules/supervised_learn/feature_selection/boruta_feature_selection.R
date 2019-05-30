suppressWarnings(library(Boruta))

args = commandArgs(trailingOnly=TRUE)

train_data_file = trimws(args[1])
test_data_file = trimws(args[2])
boruta_figs_dir = trimws(args[3])
boruta_tables_dir = trimws(args[4])
tmp_run_id = trimws(args[5])


train_df = read.table(train_data_file, sep=',', row.names=NULL, header=T)
test_df = read.table(test_data_file, sep=',', row.names=NULL, header=T)
print(dim(train_df))
print(dim(test_df))

# cleanup input files
#file.remove(train_data_file)
#file.remove(test_data_file)

# Combine train and test dfs for boruta feature selection
full_df = rbind(train_df, test_df)
print(dim(full_df))

full_df$Gene_Name = NULL

# Run Boruta algorithm
boruta <- Boruta(known_gene ~ ., data = full_df, doTrace = 1)

boruta$finalDecision = factor(boruta$finalDecision, levels(boruta$finalDecision)[c(2,1,3)])

# boruta_res = data.frame(boruta$finalDecision, 'aux'=0)
# sorted_boruta_res = boruta_res[ with(boruta_res, order(boruta.finalDecision)), ]
# sorted_boruta_res$aux = NULL


confirmed_features = getSelectedAttributes(boruta, withTentative=F)
confirmed_and_tentative = getSelectedAttributes(boruta, withTentative=T)
tentative_features = setdiff(confirmed_and_tentative, confirmed_features)

write(confirmed_features, paste(boruta_tables_dir, '/out/confirmed_features.', tmp_run_id, '.txt', sep=''))
write(tentative_features, paste(boruta_tables_dir, '/out/tentative_features.', tmp_run_id, '.txt', sep=''))


# ============
df = as.data.frame(boruta$ImpHistory)
df = do.call(data.frame, lapply(df, function(x) replace(x, is.infinite(x), NA)));
medians = apply(df, 2, median, na.rm=T);
df = df[ , order(medians)];

write.table(df, paste(boruta_tables_dir, '/out/boruta_imp_df.', tmp_run_id, '.txt',sep=''), row.names=F, quote=F)


# pdf(paste(boruta_figs_dir, 'Boruta_Feature_Importance_scores.pdf', sep='/'))
# options(repr.plot.width=9, repr.plot.height=9)
# par(mar=c(14,2,2,2))
#
# boxplot(df, outcex=0.5, outpch=8, las=2, cex.axis=0.4)
# plot(boruta, colCode = c("#41ab5d", "yellow", "#e31a1c", "#2171b5"), las=2, cex.axis=0.5, xlab='', outcex=0.4, pch=8)
# dev.off()
